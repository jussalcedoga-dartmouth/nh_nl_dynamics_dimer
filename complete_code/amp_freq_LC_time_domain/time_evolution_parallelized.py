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

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

def f(phi):
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

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

def func(t, alpha, phase, net_gain):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    d_alpha1 = -(1j*(omega1 - omega1) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase)*alpha2_c
    d_alpha2 = -(1j*(omega2 - omega2) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

# Function to solve ODE and plot the results
def solve_and_plot(args):
    net_gain, phase = args
    y0 = [1.0e7, 0.0, 1.0e7, 0.0]
    t_span = (0, 10000 * (1 / kappa_c))

    sol = solve_ivp(
        func,
        t_span, 
        y0, 
        args=(phase, net_gain), 
        method='RK45',
        dense_output=True,
        atol=1e-6,
        rtol=1e-3
    )

    t = np.linspace(t_span[0], t_span[1], 10000)
    y = sol.sol(t)

    norm = np.sqrt(y[2]**2 + y[3]**2)**2
    final_norm_value = norm[-1]

    # Define the plot folder based on net_gain
    plot_folder = f'time_evolution_with_f/undriven_time_domain_net_gain_{net_gain:.2f}'
    os.makedirs(plot_folder, exist_ok=True)

    # Plotting results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Extract parts of the signal
    alpha2_real = y[2]
    alpha2_imag = y[3]
    alpha2_complex = alpha2_real + 1j * alpha2_imag

    # Time evolution plot
    axs[0].plot(t*1e6, alpha2_real, label=r'$|\alpha_2|^2 real$', lw=2.0)
    axs[0].plot(t*1e6, alpha2_imag, label=r'$|\alpha_2|^2 imag$', lw=2.0)
    axs[0].set_xlabel(r'$t \ [\mu s]$', fontsize=20)
    axs[0].set_ylabel(r'$\alpha_2$', fontsize=20)
    axs[0].legend(loc='best', fontsize=20)

    ### extract frequency of the limit cycle
    # Slice the signal to ignore the first and last 20%
    start_index = int(0.2 * len(t))
    t = t[start_index:]
    signal = alpha2_complex[start_index:]

    # Compute the Fast Fourier Transform (FFT) of the windowed signal
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))

    # Compute the magnitudes of the FFT results
    fft_magnitude = np.abs(fft_result)

    # Find the index of the maximum amplitude in the FFT magnitude array
    max_index = np.argmax(fft_magnitude)
    # Retrieve the frequency corresponding to the maximum amplitude
    freq_LC = fft_freq[max_index]

    # Retrieve the frequency corresponding to the maximum amplitude
    freq_LC = fft_freq[max_index]
    photons_alpha_2 = fft_magnitude[max_index]/len(fft_magnitude)
    photon_numbers = photons_alpha_2.real ** 2 + photons_alpha_2.imag**2

    ## add this condition that I discussed with Michiel
    if photon_numbers <= 0.01 * alpha_sat:
        angular_freq_LC = 0 ### Setting the frequency of the oscillation to zero in the stable regime
    else:
        ## The frequency returned by the FFT corresponds to the linear frequency, and we need to translate to the angular frequency
        angular_freq_LC = 2*np.pi*freq_LC

    axs[1].plot(fft_freq, fft_result, label=r'FFT $\alpha_2$', color='royalblue', lw=2.0)
    axs[1].set_xlabel(r'Freq', fontsize=20)
    axs[1].set_ylabel(r'FFT', fontsize=20)
    axs[1].legend(loc='best', fontsize=20)

    print("Dominant frequency:", angular_freq_LC)

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/plot_phase_{phase:.2f}.png')
    plt.close()

    return [net_gain, phase, final_norm_value, angular_freq_LC]

def main():
    net_gains = np.linspace(4, 8.4, 80)
    phases = np.linspace(0, 2 * np.pi, 80)

    tasks = [(gain, phase) for gain in net_gains for phase in phases]

    results = []
    with ProcessPoolExecutor(max_workers=50) as executor:
        for result in tqdm(executor.map(solve_and_plot, tasks), total=len(tasks)):
            results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results, columns=['net_gain', 'phase', 'amplitude_lc', 'frequency_lc'])
    df.to_csv('time_evolution_with_f/results.csv', index=False)

if __name__ == '__main__':
    main()
