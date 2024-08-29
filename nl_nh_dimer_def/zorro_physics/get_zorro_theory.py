import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

with open('params.json', 'r') as f:
    parameters = json.load(f)

omega1 = parameters["omega1"] * 1e9
omega2 = parameters["omega2"] * 1e9
kappa_drive = parameters["kappa_drive"] * 1e6
kappa_readout = parameters["kappa_readout"] * 1e6
kappa_int_1 = parameters["kappa_int_1"] * 1e6
kappa_int_2 = parameters["kappa_int_2"] * 1e6
kappa_c = parameters["kappa_c"] * 1e6
beta = parameters["beta"] * 1e6
reflections_amp = parameters["reflections_amp"]
J0 = parameters["J0"] * 1e6

## Determine the drive-strenght for the Zorro experiments
epsilon_dBm = 4

## Define data folder structure to save the time and frequency domain results
data_folder_time = f'data/epsilon_{epsilon_dBm}_dBm/time'
os.makedirs(data_folder_time, exist_ok=True)

data_folder_freq = f'data/epsilon_{epsilon_dBm}_dBm/freq'
os.makedirs(data_folder_freq, exist_ok=True)

### Baseline dissipation rates, as written in the notes
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

## Expression for our on-diagonal dissipation rates
def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

save_plots = False

def f(phi):
    return 1j*J0*np.cos(phi/2)*np.exp(1j*phi/2)

### Gain of the amplifier in linear operation
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c) # alpha_sat
assert h_bar * omega1 * alpha_sat * kappa_c == P_sat

## implementation for J_nl as it is written in the notes.
def J_nl(alpha, net_gain):
    prefactor = kappa_c * 10 ** (net_gain/20)
    if alpha <= alpha_sat:
        return prefactor * 1.0
    else:
        numerator = b_amp + h_bar * omega2 * alpha_sat * kappa_c
        denominator = b_amp + h_bar * omega2 * alpha * kappa_c
        return prefactor * (numerator/denominator)

### Define every term in light of our equations of motion, as written in the notes.
def func(t, alpha, phase, net_gain, omega_d):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = alpha1_c.real**2 + alpha1_c.imag**2
    N2 = alpha2_c.real**2 + alpha2_c.imag**2

    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    d_alpha1 = -(1j*(omega_c - omega_d) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase) * alpha2_c + epsilon
    d_alpha2 = -(1j*(omega_c - omega_d) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

# Function to solve ODE and plot the results
def solve_and_plot(args):
    net_gain, phase, omega_d = args

    ## initial conditions
    y0 = [1.0e7, 0.0, 1.0e7, 0.0]

    ## Time evolve the system for 10000 * 1/kappa_c to ensure we get enough samples for FFT analysis
    t_span = (0, 10000 * (1 / kappa_c))

    ## Just use RK45 (Standard Runge-Kutta) method to achieve our numerical solutions
    sol = solve_ivp(
        func,
        t_span, 
        y0, 
        args=(phase, net_gain, omega_d), 
        method='RK45',
        dense_output=True,
        atol=1e-6,
        rtol=1e-3
    )

    ## Determine the number of t-samples for defininf the sampling rate for FFT analysis
    t = np.linspace(t_span[0], t_span[1], 10000)
    y = sol.sol(t)

    # Extract parts of the signal
    alpha2_real = y[2]
    alpha2_imag = y[3]
    alpha2_complex = alpha2_real + 1j * alpha2_imag

    ### Ditch the first 20% of the signal to ignore transient behavior
    start_index = int(0.2 * len(t))
    t = t[start_index:]
    signal = alpha2_complex[start_index:]

    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))

    # Normalize FFT to get correct amplitude representation
    fft_norm = fft_result / len(signal)  # Normalize by number of samples

    # Calculate power spectral density to obtain photon numbers across the spectrum
    psd = fft_norm.real ** 2 +  fft_norm.imag ** 2

    # Apply fftshift to align zero frequency at center
    ## fftshift, just translates the regular fft spectrum so that the 0.0 frequency is at the center.
    ### the output of fft usually puts the 0.0 frequency in the first element.
    fft_freq_shifted = np.fft.fftshift(fft_freq) * 2 * np.pi  # Convert to angular freq.
    psd_shifted = np.fft.fftshift(psd)

    ## assuming power spectral density corresponds to photon numbers in this setting.
    power_watts = psd_shifted * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)

    # Save time-domain results to CSV for visualization
    time_traces = pd.DataFrame({
        'time': t,
        'alpha_2_real': signal.real,
        'alpha_2_imag': signal.imag
    })

    time_traces.to_csv(f'{data_folder_time}/fft_results_omega_d_{omega_d/1e9:.4f}_GHz.csv', index=False)

    # Save FFT results to CSV for visualization
    df_fft = pd.DataFrame({
        'FFT Frequency [Hz]': fft_freq_shifted,
        'FFT Magnitude': power_dBm
    })

    df_fft.to_csv(f'{data_folder_freq}/fft_results_omega_d_{omega_d/1e9:.4f}_GHz.csv', index=False)

phase = np.pi ## define the phase
omega_c = omega1 ## Define the resonance frequency of the oscillators
net_gain = 8.4 ## Select a net gain value at which we park at

## Determine the range of frequencies for the sweep. I'm using the exact same span that I used in the experiments.
frequencies = np.linspace(omega_c - 8e6, omega_c + 8e6, 160)

### Prepare the tasks indexed by the corresponding omega_d value
tasks = [(net_gain, phase, omega_d) for omega_d in frequencies]

# Execute tasks concurrently parallelizing the data collection in max_workers cores.
with ProcessPoolExecutor(max_workers=50) as executor:
    list(tqdm(executor.map(solve_and_plot, tasks), total=len(tasks)))
