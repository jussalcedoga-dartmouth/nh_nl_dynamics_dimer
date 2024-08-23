import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import windows, butter, filtfilt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
from tqdm import tqdm  # Import tqdm for the progress bar functionality
import pandas as pd

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

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

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34
epsilon_dBm = -30 # dBm

def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

def f(phi):
    # return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)
    ## Let's get rid of the squared term...
    return 1j*J0*np.cos(phi/2)*np.exp(1j*phi/2)

### Gain of the amplifier in linear operation
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c) # alpha_sat
assert h_bar * omega1 * alpha_sat * kappa_c == P_sat

## implementation for J_nl as it is written in the notes.
def J_nl(alpha, net_gain):
    if alpha <= alpha_sat:
        return kappa_c * 10 ** (net_gain/20)
    else:
        numerator = b_amp + h_bar * omega2 * alpha_sat * kappa_c
        denominator = b_amp + h_bar * omega2 * alpha * kappa_c
        return kappa_c * 10 ** (net_gain/20) * (numerator/denominator)

## Equations of motion
def func(t, alpha, omega_d, phase, net_gain, epsilon_dBm):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))

    ## run this without the sqrt.
    N1 = alpha1_c.real**2 + alpha1_c.imag**2
    N2 = alpha2_c.real**2 + alpha2_c.imag**2

    ## As written in the notes
    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    d_alpha1 = -(1j*(omega1 - omega_d) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase)*alpha2_c + epsilon
    d_alpha2 = -(1j*(omega2 - omega_d) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

# Constants
h_bar = 1.054571817e-34
epsilon_dBm = -30  # dBm
phase = np.pi
# phase = 0.0

### Simpler implementation, without the filtering
def simulate_frequency(args):
    omega_d, net_gain = args

    # Initial conditions. They are fixed for now, but we can try different (or even random) ones
    y0 = [1.0e7, 0.0, 1.0e7, 0.0]

    # Time span is set so that we time evolve for 30 times the characteristic dissipation rate kappa_c
    t_span = (0, 10000*(1/kappa_c))

    # Solve the ODE
    sol = solve_ivp(
        func,
        t_span, 
        y0, 
        args=(omega_d, phase, net_gain, epsilon_dBm), 
        method='RK45',
        dense_output=True,
        atol=1e-6,
        rtol=1e-3
    )

    #### plotting.
    t = np.linspace(t_span[0], t_span[1], 100000)
    y = sol.sol(t)

    # Extract parts of the signal
    alpha2_real = y[2]
    alpha2_imag = y[3]
    alpha2_complex = alpha2_real + 1j * alpha2_imag

    ### This is a very clean implementation!!!!
    start_index = int(0.2 * len(t))
    t = t[start_index:]
    signal = alpha2_complex[start_index:]

    # Compute the Fast Fourier Transform (FFT) of the windowed signal
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))

    # Isolate the DC component in the frequency domain
    dc_fft = np.zeros_like(fft_result)
    dc_index = np.where(fft_freq == 0.0)[0]
    dc_fft[dc_index] = fft_result[dc_index]

    ### Photon numbers from the DC component
    photon_numbers = dc_fft[0]/len(dc_fft)
    photon_dc_component = photon_numbers.real ** 2 + photon_numbers.imag**2
    power_watts = photon_dc_component * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)
    power_dB = power_dBm - epsilon_dBm

    return omega_d, power_dB

def run_simulation_for_gain(net_gain):
    # Define the frequency range
    frequencies = np.linspace(5.975e9, 6.085e9, 1000)
    results = []
    
    # Folder setup
    gain_folder = f'single_traces_nonhermitian/data/'
    os.makedirs(gain_folder, exist_ok=True)
    csv_path = os.path.join(gain_folder, f'results_gain_{net_gain:.2f}.csv')
    
    # Folder setup
    gain_folder_images = f'single_traces_nonhermitian/images/'
    os.makedirs(gain_folder_images, exist_ok=True)
    png_path = os.path.join(gain_folder_images, f'plot_gain_{net_gain:.2f}.png')

    # Run parallel simulation
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(simulate_frequency, (omega_d, net_gain)) for omega_d in frequencies]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Gain {net_gain:.2f}"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results, columns=['Frequency', 'Power_dB'])
    df.sort_values('Frequency', inplace=True)  # Sort by frequency
    df.to_csv(csv_path, index=False)
    
    # Plotting each gain result
    plt.figure(figsize=(7,7))
    plt.plot(df['Frequency'] / 1e9, df['Power_dB'])
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(r'$S_{21}$ [dB]')
    plt.title(f'Results for Gain {net_gain:.2f}')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

# List of gains
gains = np.linspace(-4.6, 8.4, 31)[::-1]  # Example gain values from -3dB to +3dB

for gain in gains:
    print(f'Running simulation for net gain: {gain:2f}')
    run_simulation_for_gain(gain)
