import numpy as np
from scipy.optimize import root, fsolve, root_scalar, least_squares
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf
from matplotlib import cm
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

### This code gets rid of any numerical data from previous executions, and only keeps the latest
def clear_folder_contents(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove the file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove subdirectories recursively

# Folders to manage
folders = [
    'numerical_data',
    'numerical_data/hermitian',
    'numerical_data/nonhermitian'
]

# Create or clear the folders
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)  # Create folder if it doesn't exist
    else:
        clear_folder_contents(folder)  # Clear the folder if it exists

## For individual numerical plots... I am mostly interested in them compared to experiments.
save_individual = False

## create folder to save the plots
folder_name = 'plots'
os.makedirs(folder_name, exist_ok=True)

## load the numerical parameters from the JSON
with open('numerical_parameters.json', 'r') as file:
    params = json.load(file)

### assign the parameters accordingly...
MHz = 1e6
GHz = 1e9
omega1 = params['omega1'] * GHz
omega2 = params['omega2'] * GHz
h_bar = params['h_bar']
t_hop_1_1 = params['t_hop_1_1'] * MHz
t_hop_1_2 = params['t_hop_1_2'] * MHz
t_hop_2_1 = params['t_hop_2_1'] * MHz
t_hop_2_2 = params['t_hop_2_2'] * MHz

readout_kappa = params['readout_kappa'] * MHz
drive_kappa = params['drive_kappa'] * MHz

kappa_0_1 = params['kappa_0_1'] * MHz
kappa_0_2 = params['kappa_0_2'] * MHz

### uncharacterized dissipation...
extra_dissipation = params['extra_dissipation'] * MHz

hermitian_num_points = params['hermitian_num_points']
nh_num_points = params['nh_num_points']

epsilon_dBm = params['epsilon_dBm']
gain_threshold = params['gain_threshold']
G_0 = params['G_0']
max_symm_gain = 9.5 ### The max gain queried directly from the hashmap
insertion_loss = G_0 - max_symm_gain
num_attenuations = params['num_attenuations']

kappa_cavity1 = kappa_0_1 + readout_kappa + t_hop_2_1 + t_hop_1_1 + extra_dissipation
kappa_cavity2 = kappa_0_2 + drive_kappa + t_hop_2_1 + t_hop_1_1 + extra_dissipation

# Function to calculate net gain from attenuation
def calculate_net_gain(attenuation, G_0=G_0, insertion_loss=insertion_loss):
    return G_0 - insertion_loss - attenuation

def dbm_to_watts(dbm):
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

gain_1 = G_0
a_1, b_1, c_1, flat_line_value_1, x0_1 = pwf.return_params(gain_1)

gain_2 = G_0
a_2, b_2, c_2, flat_line_value_2, x0_2 = pwf.return_params(gain_2)

def piece_wise_amp_function(x, a, b, c, flat_line_value, x0):
    return np.where(x <= x0, flat_line_value, model_function(x, a, b, c))

def dBm_to_dB(S21_dBm, input_power_dBm=-10):
    relative_power_dBm = S21_dBm
    relative_power_dB = relative_power_dBm - input_power_dBm
    return relative_power_dB

def func(alpha, omega_d, phase, attenuation, epsilon_dBm = -10):

    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((drive_kappa * epsilon_watts) / (h_bar * omega_d))

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * t_hop_1_1/4
    N2_watts = h_bar * omega2 * N2 * t_hop_1_2/4

    G1 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * 10 ** (-(attenuation + insertion_loss)/20)
    G2 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * 10 ** (-(attenuation + insertion_loss)/20)

    d_alpha1 = -(((kappa_cavity1)/2 - (G2)*np.sqrt(t_hop_1_1 * t_hop_2_1)/2) + 1j * (omega1 - omega_d)) * alpha1_c + 1j * G2 * np.sqrt(t_hop_1_1 * t_hop_2_1) * alpha2_c
    d_alpha2 = -(((kappa_cavity2)/2 - (G1)*np.sqrt(t_hop_1_2 * t_hop_2_2)*np.exp(-1j * phase)/2) + 1j * (omega2 - omega_d)) * alpha2_c + 1j * G1 * np.sqrt(t_hop_1_2 * t_hop_2_2) * np.exp(-1j * phase) * alpha1_c + epsilon

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

def fixed_points(omega_d_val, phase_val, att, initial_guess, epsilon_dBm=-10):

    if phase_val == 0:
        tolerance = 1e-8
        sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='anderson')
        if sol.success:
            return sol.x[0] + 1j * sol.x[1], sol.x[2] + 1j * sol.x[3]
        else:
            return None, None
    else:
        tolerance = 1e-4
        sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='hybr')
        if sol.success:
            return sol.x[0] + 1j * sol.x[1], sol.x[2] + 1j * sol.x[3]
        else:
            return None, None

def calculate_power(N, kappa, hbar, omega):
    return N * kappa * hbar * omega/4

# Function to convert power in Watts to dBm
def power_to_dBm(power_watts):
    return 10 * np.log10(power_watts * 1e3)

### read file with optimized parameters....
def create_plots_for_phase(frequencies, attenuations, phase, epsilon_dBm=-10):
    alpha2_solutions = np.empty((len(frequencies), len(attenuations)))
    
    # Initial guesses can depend on the phase; they could also be dynamically adjusted
    initial_guess = [1e6, 1e6, 1e6, 1e6] if phase == 0.0 else [1e8, 1e8, 1e8, 1e8]

    for j, attenuation in enumerate(attenuations):
        # last_successful_guess = initial_guess  # Start with the predefined initial guess

        for i, omega_d_val in enumerate(frequencies):
            try:
                # Use last successful guess as the initial condition
                alpha1_sol, _ = fixed_points(omega_d_val, phase, attenuation, initial_guess, epsilon_dBm=epsilon_dBm)

                # if alpha1_sol is not None:
                #     # If the fixed point solver was successful, update the last_successful_guess
                #     last_successful_guess = [alpha1_sol.real, alpha1_sol.imag, alpha1_sol.real, alpha1_sol.imag]  # Update with the real parts of the solution

                ## Compute photon numbers and convert it to a dB scale to compare with experiment
                N1_total = np.sqrt(alpha1_sol.real**2 + alpha1_sol.imag**2)**2
                N2_watts = calculate_power(N1_total, readout_kappa, h_bar, omega1)  # Readout cavity 1
                N2_dBm = power_to_dBm(N2_watts)
                N2_dB = dBm_to_dB(N2_dBm)
                alpha2_solutions[i, j] = N2_dB

            except Exception as e:
                # In case of any error during the computation, log it or handle it appropriately
                # print(f"Error at freq: {omega_d_val}, att: {attenuation}, error: {str(e)}")
                alpha2_solutions[i, j] = -25  # Set a default or error value
                # last_successful_guess = initial_guess

    return alpha2_solutions

### Hermitian results
frequencies = np.linspace(5.935e9, 6.135e9, hermitian_num_points)
attenuations = np.linspace(0.0, 14.5, num_attenuations)
phase = 0
alpha2_solutions = create_plots_for_phase(frequencies, attenuations, phase, epsilon_dBm=epsilon_dBm)

center_frequency = np.mean([omega1, omega2])/1e9
frequency_array = frequencies / 1e9  # Convert to GHz for plotting

# Placeholder for results
hermitian_results = {
    "left_peak_freqs": [],
    "right_peak_freqs": [],
    "peak_distances": [],
    "max_transmission_left": [],
    "max_transmission_right": []
}

net_gains = calculate_net_gain(np.array(attenuations))

for i, attenuation in enumerate(attenuations):
    # Slice for transmissions below center frequency
    below_center = alpha2_solutions[:, i][frequency_array < center_frequency]
    left_peak_index = np.argmax(below_center)
    left_peak_freq = frequency_array[frequency_array < center_frequency][left_peak_index]
    max_trans_left = below_center[left_peak_index]

    # Slice for transmissions above center frequency
    above_center = alpha2_solutions[:, i][frequency_array > center_frequency]
    right_peak_index = np.argmax(above_center)
    right_peak_freq = frequency_array[frequency_array > center_frequency][right_peak_index]
    max_trans_right = above_center[right_peak_index]

    # Store results
    hermitian_results["left_peak_freqs"].append(left_peak_freq)
    hermitian_results["right_peak_freqs"].append(right_peak_freq)
    hermitian_results["peak_distances"].append((right_peak_freq - left_peak_freq)*1e3)
    hermitian_results["max_transmission_left"].append(max_trans_left)
    hermitian_results["max_transmission_right"].append(max_trans_right)

if save_individual:
    # Hermitian phase plots
    plt.figure(figsize=(15, 4))

    # Left and Right Peak Frequencies
    plt.subplot(1, 3, 1)
    plt.plot(net_gains, hermitian_results['left_peak_freqs'], 'ro-', label='Left Peak Frequency')
    plt.plot(net_gains, hermitian_results['right_peak_freqs'], 'bo-', label='Right Peak Frequency')
    plt.axvline(x=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Frequency [GHz]')
    plt.title('Hermitian Phase: Peak Frequencies')
    plt.legend()

    # Peak Distance
    plt.subplot(1, 3, 2)
    plt.plot(net_gains, hermitian_results['peak_distances'], 'go-', label='Peak Distance')
    plt.axvline(x=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Distance [MHz]')
    plt.title('Hermitian Phase: Peak Distance')
    plt.legend()

    # Max Transmission for Peaks
    plt.subplot(1, 3, 3)
    plt.plot(net_gains, hermitian_results['max_transmission_left'], 'ro-', label='Max Transmission Left Peak')
    plt.plot(net_gains, hermitian_results['max_transmission_right'], 'bo-', label='Max Transmission Right Peak')
    plt.axvline(x=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Transmission [dB]')
    plt.title('Hermitian Phase: Max Transmission for Peaks')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/hermitian_numerical_results.png')
    plt.close()
else:
    pass

### Now, non-Hermitian stuff
frequencies_nh = np.linspace(6.012e9, 6.047e9, nh_num_points)
attenuations = np.linspace(0.0, 14.5, num_attenuations)

phase = np.pi
alpha2_solutions = create_plots_for_phase(frequencies_nh, attenuations, phase, epsilon_dBm=epsilon_dBm)

center_frequency = np.mean([omega1, omega2])/1e9
frequency_array = frequencies / 1e9  # Convert to GHz for plotting

# For the non-Hermitian phase, you'd similarly find the overall max transmission and its frequency:
non_hermitian_results = {
    "max_transmission_freqs": [],
    "max_transmission_values": []
}

for i, attenuation in enumerate(attenuations):
    max_trans_index = np.argmax(alpha2_solutions[:, i])
    max_trans_freq = frequency_array[max_trans_index]
    max_trans_value = alpha2_solutions[:, i][max_trans_index]

    # Store results
    non_hermitian_results["max_transmission_freqs"].append(max_trans_freq)
    non_hermitian_results["max_transmission_values"].append(max_trans_value)

if save_individual:
    # Non-Hermitian phase plots
    plt.figure(figsize=(10, 4))

    # Frequency of Maximum Transmission
    plt.subplot(1, 2, 1)
    plt.plot(net_gains, non_hermitian_results['max_transmission_freqs'], 'ro-', label='Frequency of Max Transmission')
    plt.axvline(x=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Frequency [GHz]')
    plt.title('Non-Hermitian Phase: Frequency of Maximum Transmission')
    plt.legend()

    # Maximum Transmission Value
    plt.subplot(1, 2, 2)
    plt.plot(net_gains, non_hermitian_results['max_transmission_values'], 'bo-', label='Maximum Transmission Value')
    plt.axvline(x=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Transmission [dB]')
    plt.title('Non-Hermitian Phase: Maximum Transmission Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/nonhermitian_numerical_results.png')
    plt.close()
else:
    pass

def save_results(file_path, results):
    with open(file_path, 'w') as f:
        json.dump(results, f)

### Now, single transmission traces
def store_numerical_traces(frequencies, attenuations, phase_alias):
    folder_name = f'numerical_data/{phase_alias}'
    os.makedirs(folder_name, exist_ok=True)

    phases = {'hermitian': 0, 'nonhermitian': np.pi}

    alpha2_solutions = create_plots_for_phase(frequencies, attenuations, phases[phase_alias], epsilon_dBm=epsilon_dBm)
    
    # Save each trace to a separate CSV file
    for i, net_gain in enumerate(net_gains):
        # Create a DataFrame
        df = pd.DataFrame({
            'Frequency': frequencies,
            'Transmission': alpha2_solutions[:, i]
        })
        # Define the file path
        file_path = f'{folder_name}/hermitian_phase_symmetric_attenuation_{net_gain}_dB.csv'
        df.to_csv(file_path, index=False)

# Call this at the end of your numerical analysis script
if __name__ == '__main__':
    hermitian_results = {
        'left_peak_freqs': hermitian_results['left_peak_freqs'],
        'right_peak_freqs': hermitian_results['right_peak_freqs'],
        'peak_distances': hermitian_results['peak_distances'],
        'max_transmission_left': hermitian_results['max_transmission_left'],
        'max_transmission_right': hermitian_results['max_transmission_right'],
        'attenuations': net_gains.tolist()
    }

    non_hermitian_results = {
        'max_transmission_freqs': non_hermitian_results["max_transmission_freqs"],
        'max_transmission_values': non_hermitian_results["max_transmission_values"],
        'attenuations': net_gains.tolist()
    }

    # Save the results to files
    save_results('numerical_data/hermitian_results.json', hermitian_results)
    save_results('numerical_data/nonhermitian_results.json', non_hermitian_results)

    # Call this function for each phase
    store_numerical_traces(frequencies, attenuations, 'hermitian')
    store_numerical_traces(frequencies_nh, attenuations, 'nonhermitian')
