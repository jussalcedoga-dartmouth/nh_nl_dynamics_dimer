import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import find_peaks

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

omega_d  = omega1 ## drive strictly at omega_c

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

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

def func(t, alpha, phase, net_gain, epsilon_dBm):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    if epsilon_dBm != "None":
        epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
        epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))
    else:
        pass

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = alpha1_c.real**2 + alpha1_c.imag**2
    N2 = alpha2_c.real**2 + alpha2_c.imag**2

    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    if epsilon_dBm == "None": ## Just for the undriven baseline contour for reference. We still need to be in the 'omega_c frame'
        d_alpha1 = -(1j*(omega1 - omega_d) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase)*alpha2_c
        d_alpha2 = -(1j*(omega2 - omega_d) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c
    else:
        d_alpha1 = -(1j*(omega1 - omega_d) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase)*alpha2_c + epsilon
        d_alpha2 = -(1j*(omega2 - omega_d) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

# Function to solve ODE and plot the results
def solve_and_plot(args):
    net_gain, phase, epsilon_dBm = args
    y0 = [1.0e7, 0.0, 1.0e7, 0.0]
    t_span = (0, 10000 * (1 / kappa_c))

    sol = solve_ivp(
        func,
        t_span, 
        y0, 
        args=(phase, net_gain, epsilon_dBm), 
        method='RK45',
        dense_output=True,
        atol=1e-6,
        rtol=1e-3
    )

    t = np.linspace(t_span[0], t_span[1], 100000)
    y = sol.sol(t)

    norm = np.sqrt(y[2]**2 + y[3]**2)**2
    final_norm_value = norm[-1]

    if save_plots:
        # Define the plot folder based on net_gain
        plot_folder = f'synchronization_epsilon_{epsilon_dBm}/undriven_time_domain_net_gain_{net_gain:.2f}'
        os.makedirs(plot_folder, exist_ok=True)
    else:
        pass

    if save_plots:
        # Plotting results
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    else:
        pass

    # Extract parts of the signal
    alpha2_real = y[2]
    alpha2_imag = y[3]
    alpha2_complex = alpha2_real + 1j * alpha2_imag

    if save_plots:
        # Time evolution plot
        axs[0].plot(t*1e6, alpha2_real, label=r'$|\alpha_2|^2 real$', lw=2.0)
        axs[0].plot(t*1e6, alpha2_imag, label=r'$|\alpha_2|^2 imag$', lw=2.0)
        axs[0].set_xlabel(r'$t \ [\mu s]$', fontsize=20)
        axs[0].set_ylabel(r'$\alpha_2$', fontsize=20)
        axs[0].legend(loc='best', fontsize=20)
    else:
        pass

    ### extract frequency of the limit cycle
    # Slice the signal to ignore the first 20%
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
    photon_numbers = photons_alpha_2.real**2 + photons_alpha_2.imag**2

    ## add this condition that I discussed with Michiel
    if photon_numbers <= 0.00001 * alpha_sat:
        angular_freq_LC = 0 ### Setting the frequency of the oscillation to zero in the stable regime
        binary_result = 0.0
    else:
        if epsilon_dBm != "None":
            ## The frequency returned by the FFT corresponds to the linear frequency, and we need to translate to the angular frequency
            angular_freq_LC = 2*np.pi*freq_LC ## convert linear to angular frequency
            peaks, _ = find_peaks(fft_magnitude, height=1e10) ### This can be better. This is just a proof of concept
            number_of_peaks = len(peaks)
            binary_result = 1 if number_of_peaks >= 2 else 0
        else:
            ### If no drive, track down the self-oscillation by itself.
            angular_freq_LC = 2*np.pi*freq_LC ## convert linear to angular frequency
            peaks, _ = find_peaks(fft_magnitude, height=1e10) ### This can be better. This is just a proof of concept
            number_of_peaks = len(peaks)
            binary_result = 1 if number_of_peaks >= 1 else 0 ## a single peak in the vacuum unstable regime, in the drivenless case means a self-oscillation. 

    if save_plots:
        axs[1].plot(2*np.pi*fft_freq/1e6, fft_magnitude, label=r'FFT $\alpha_2$', color='royalblue', lw=2.0)
        axs[1].set_xlabel(r'Freq', fontsize=20)
        axs[1].set_ylabel(r'FFT', fontsize=20)
        axs[1].legend(loc='best', fontsize=20)

        plt.tight_layout()
        plt.savefig(f'{plot_folder}/plot_phase_{phase:.2f}.png')
        plt.close()
    else:
        pass

    print("Dominant frequency:", angular_freq_LC, "SO-Drive?", binary_result)

    return [net_gain, phase, final_norm_value, angular_freq_LC, binary_result]

def process_epsilon(epsilon_dBm):
    net_gains = np.linspace(4, 8.4, 60)
    phases = np.linspace(0, 2 * np.pi, 80)
    
    # Prepare tasks with the current epsilon
    tasks = [(gain, phase, epsilon_dBm) for gain in net_gains for phase in phases]
    
    results = []
    with ProcessPoolExecutor(max_workers=50) as executor:
        # Map tasks to the executor and progress bar
        for result in tqdm(executor.map(solve_and_plot, tasks), total=len(tasks)):
            results.append(result)
    
    # Save results to CSV
    df = pd.DataFrame(results, columns=['Gain [dB]', 'Phase [rads.]', 'Amplitude [dBm]', 'Frequency [Hz]', 'Flag'])
    df.to_csv(f'better_resolution/results_epsilon_{epsilon_dBm}_dBm.csv', index=False)

def main():
    
    # Epsilons to process
    epsilons = ['None', 0, 4, 8, 12, 16]

    # Process each epsilon value separately
    for epsilon in epsilons:
        print(epsilon)
        process_epsilon(epsilon)

if __name__ == '__main__':
    main()
