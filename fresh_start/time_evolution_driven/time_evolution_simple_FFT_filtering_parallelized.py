import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import windows, butter, filtfilt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
from tqdm import tqdm  # Import tqdm for the progress bar functionality

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
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

### Gain of the amplifier in linear operation
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c) # alpha_sat
print(h_bar * omega1 * alpha_sat * kappa_c)
print(P_sat)
assert h_bar * omega1 * alpha_sat * kappa_c == P_sat

## implementation for J_nl as it is written in the notes.
def J_nl(alpha, net_gain):
    if alpha <= alpha_sat:
        return kappa_c * 10 ** (net_gain/20)
    else:
        numerator = b_amp + h_bar * omega2 * alpha_sat * kappa_c
        denominator = b_amp + h_bar * omega2 * alpha * kappa_c
        return kappa_c * 10 ** (net_gain/20) * (numerator/denominator)

def func(t, alpha, omega_d, phase, net_gain, epsilon_dBm):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

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

net_gain = 8.4

def simulate_frequency(omega_d):

    # A = matrix_A(net_gain, phase, omega_d)
    # eigenvalues = np.linalg.eigvals(A)
    # stability = np.max(eigenvalues.real)

    # if stability > 0:
    # print('Unstable')
    # Initial conditions. They are fixed for now, but we can try different (or even random) ones
    y0 = [1.0e7, 0.0, 1.0e7, 0.0]

    # Time span is set so that we time evolve for 30 times the characteristic dissipation rate kappa_c
    t_span = (0, 1000*(1/kappa_c))

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

    # Slice the signal to ignore the first and last 20%
    start_index = int(0.2 * len(t))
    end_index = int(0.8 * len(t))
    t = t[start_index:end_index]
    signal = alpha2_complex[start_index:end_index]

    # if windowed:
    # Apply a window function to the sliced signal
    window = windows.hann(len(signal))
    signal_windowed = signal * window

    # Compute the Fast Fourier Transform (FFT) of the windowed signal
    fft_result = np.fft.fft(signal_windowed)
    fft_freq = np.fft.fftfreq(len(signal_windowed), d=(t[1] - t[0]))

    # fft_result = np.fft.fft(signal)
    # fft_freq = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))

    # Apply low-pass filter
    # cutoff_frequency = 1e6  # Define cutoff frequency as needed

    # Sample rate and desired cutoff frequency of the filter
    fs = 1 / (t[1] - t[0])  # Sampling frequency
    cutoff_frequency = 0.1 * fs  # Cutoff frequency as a fraction of the sampling rate
    # cutoff_frequency = 30e6  # Cutoff frequency as a fraction of the sampling rate

    b, a = butter(1, cutoff_frequency, 'low', fs=1/(t[1] - t[0]))
    fft_result_filtered = filtfilt(b, a, fft_result)

    # Isolate the DC component in the frequency domain
    dc_fft = np.zeros_like(fft_result_filtered)
    dc_index = np.where(fft_freq == 0.0)[0]
    dc_fft[dc_index] = fft_result[dc_index]

    # Perform Inverse FFT to get the DC component in the time domain
    # dc_time = np.fft.ifft(dc_fft).real
    # dc_time = np.fft.ifft(dc_fft).real + 1j * np.fft.ifft(dc_fft).imag

    # Compute the dB scale for power
    # photon_dc_component = np.abs(dc_time)**2
    photon_dc_component = np.sqrt(np.fft.ifft(dc_fft).real**2 + np.fft.ifft(dc_fft).imag**2)**2
    power_watts = photon_dc_component * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)
    power_dB = list(set(power_dBm - epsilon_dBm))[0]
    # power_dB = power_dBm - epsilon_dBm
    # print(power_dB)
    return omega_d, power_dB

    # else:
    #     print('Stable')
    #     omega_d, power_dB = transmission_linear(omega_d, net_gain, phase, epsilon_dBm)

    #     return omega_d, power_dB

# List of frequencies
frequencies = np.linspace(5.975e9, 6.085e9, 1000)

# Using ProcessPoolExecutor to parallelize the simulation across multiple cores
results = []
with ProcessPoolExecutor(max_workers=50) as executor:
    # Submit all tasks and get a dictionary of future objects
    futures = {executor.submit(simulate_frequency, omega_d): omega_d for omega_d in frequencies}
    
    # Iterate over the completed tasks using as_completed, which yields futures as they complete
    for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Frequencies"):
        omega_d = futures[future]
        try:
            result = future.result()
            results.append(result)  # Collect results after they are completed
        except Exception as e:
            print(f"Task with frequency {omega_d/1e9} GHz failed: {e}")

# Sorting results by frequency
results.sort(key=lambda x: x[0])
frequencies, transmission_dB = zip(*results)

# Plotting results
plt.figure(figsize=(7, 7))
plt.plot(np.array(frequencies) / 1e9, transmission_dB, lw=2.0)
plt.xlabel('Frequency [GHz]')
plt.ylabel(r'$S_{21}$ [dB]')
plt.tight_layout()
plt.savefig(f'steady_state_window_function_parallelized_net_gain_{net_gain}_no_window.png')
plt.close()
