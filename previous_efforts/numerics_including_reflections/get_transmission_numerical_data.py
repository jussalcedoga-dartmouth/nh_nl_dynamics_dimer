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
from collections import defaultdict

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

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
kappa_c = params['kappa_c'] * MHz

readout_kappa = params['readout_kappa'] * MHz
drive_kappa = params['drive_kappa'] * MHz

kappa_0_1 = params['kappa_0_1'] * MHz
kappa_0_2 = params['kappa_0_2'] * MHz

hermitian_num_points = params['hermitian_num_points']
nh_num_points = params['nh_num_points']

epsilon_dBm = params['epsilon_dBm']
gain_threshold = params['gain_threshold']
G_0 = params['G_0']

max_symm_gain = 9.5
insertion_loss = G_0 - max_symm_gain

num_attenuations = params['num_attenuations']

# Function to calculate net gain from attenuation
def calculate_net_gain(attenuation, G_0=G_0, insertion_loss=insertion_loss):
    return G_0 - insertion_loss - attenuation

def dbm_to_watts(dbm):
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

def model_function_reflection(x, a, b, c):
    return -a * (1 / (1 + (x / b))) + c

## fitted values from transmission and reflection from the amp - with no c
gain_1 = G_0
a_1, b_1, c_1, flat_line_value_1, x0_1 =  9.2448, 0.008638, 0.0, 8.2716171, 0.0007981

## transmission
gain_2 = G_0
a_2, b_2, c_2, flat_line_value_2, x0_2 =  9.2448, 0.008638, 0.0, 8.2716171, 0.0007981

## reflections off the amp
a_r, b_r, c_r, flat_line_r, x0_r = 0.3554, 0.00634, 0.73626, 0.4854, 0.0007981

def piece_wise_amp_function(x, a, b, c, flat_line_value, x0):
    return np.where(x <= x0 + 0.0002, flat_line_value, model_function(x, a, b, c))

def piece_wise_amp_reflection(x_r, a_r, b_r, c_r, flat_line_r, x0_r):
    return np.where(x_r <= x0_r + 0.002, flat_line_r, model_function_reflection(x_r, a_r, b_r, c_r))

def dBm_to_dB(S21_dBm, input_power_dBm=-10):
    relative_power_dBm = S21_dBm
    relative_power_dB = relative_power_dBm - input_power_dBm
    return relative_power_dB

def find_closest_gain_parameters(df, gain_value):
    # Find the index of the row with the closest gain value
    closest_idx = (df['gain'] - gain_value).abs().idxmin()
    
    # Extract the parameters for the closest gain
    closest_params = df.loc[closest_idx, ['slope', 'intercept']]
    slope = closest_params['slope']
    intercept = closest_params['intercept']

    return slope, intercept

def find_closest_parameters(df, phase_value):
    # Find the index of the row with the closest phase value
    closest_idx = (df['phase'] - phase_value).abs().idxmin()

    # Find the index of the row with the closest new phase value    
    # Extract the parameters for the closest new phase
    closest_params = df.loc[closest_idx, ['slope', 'intercept']]
    slope = closest_params['slope']
    intercept = closest_params['intercept']

    return slope, intercept

def model_function_reflection_ps_da(x, slope, intercept):
    return slope * x + intercept

csv_path_da = 'csvs_calibrations/linear_fit_parameters_by_gain.csv'
df_da = pd.read_csv(csv_path_da)

csv_path = 'csvs_calibrations/linear_fit_parameters_by_phase.csv'
df = pd.read_csv(csv_path)

def fixed_points(omega_d_val, phase_val, att, initial_guess, epsilon_dBm=-10):
    if phase_val == 0:
        tolerance = 1e-8
        sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='hybr')
        if sol.success:
            return sol.x[0] + 1j * sol.x[1], sol.x[2] + 1j * sol.x[3]
        else:
            return None, None
    else:
        tolerance = 1e-5
        sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='hybr')
        if sol.success:
            return sol.x[0] + 1j * sol.x[1], sol.x[2] + 1j * sol.x[3]
        else:
            return None, None

def func(alpha, omega_d, phase, attenuation, epsilon_dBm = -10):

    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((drive_kappa * epsilon_watts) / (h_bar * omega_d))

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * kappa_c/4
    N2_watts = h_bar * omega2 * N2 * kappa_c/4

    eta_A = 10**(-attenuation/20)
    eta_I = 10**(-insertion_loss/20)

    nu_G12 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * eta_A * eta_I
    nu_G21 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * eta_A * eta_I

    nu_r1 = piece_wise_amp_reflection(N1_watts, a_r, b_r, c_r, flat_line_r, x0_r)
    nu_r2 = piece_wise_amp_reflection(N2_watts, a_r, b_r, c_r, flat_line_r, x0_r)

    ## Reflections off the phase shifter
    slope_ps, intercept_ps = find_closest_parameters(df, phase)
    nu_ps = model_function_reflection_ps_da(N1_watts, slope_ps, intercept_ps)

    ## Reflections off the attenuator
    query_gain_value = max_symm_gain - attenuation
    slope_da, intercept_da = find_closest_gain_parameters(df_da, query_gain_value)
    nu_da = model_function_reflection_ps_da(N2_watts, slope_da, intercept_da)

    if nu_G12 >= 1.0:
        kappa_diag_1 = kappa_0_1 + drive_kappa + kappa_c - (nu_r1 + nu_ps)*kappa_c - (nu_G21 - 1)*kappa_c
        kappa_diag_2 = kappa_0_2 + readout_kappa + kappa_c - (nu_r2 + nu_da)*kappa_c - (nu_G12 - 1)*kappa_c
    else:
        kappa_diag_1 = kappa_0_1 + drive_kappa + kappa_c - (nu_r1 + nu_ps)*kappa_c
        kappa_diag_2 = kappa_0_2 + readout_kappa + kappa_c - (nu_r2 + nu_da)*kappa_c

    print('\n', kappa_diag_1/1e6, kappa_diag_2/1e6, ', gain: ', query_gain_value, ', nu_da: ', nu_da, ', nu_r1: ', nu_r1, ', nu_r2: ', nu_r2, ', nu_ps: ', nu_ps)

    d_alpha1 = -(kappa_diag_1 + 1j*(omega1 - omega_d))*alpha1_c \
                - 1j * nu_G21*np.exp(1j*(phase))*kappa_c*alpha2_c + epsilon

    d_alpha2 = -(kappa_diag_2 + 1j*(omega2 - omega_d))*alpha2_c \
                - 1j * nu_G12*kappa_c*alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

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

        for i, omega_d_val in enumerate(frequencies):
            try:
                # Use last successful guess as the initial condition
                _, alpha2_sol = fixed_points(omega_d_val, phase, attenuation, initial_guess, epsilon_dBm=epsilon_dBm)

                ## Compute photon numbers and convert it to a dB scale to compare with experiment
                N2_total = np.sqrt(alpha2_sol.real**2 + alpha2_sol.imag**2)**2
                N2_watts = calculate_power(N2_total, readout_kappa, h_bar, omega2)  # Readout cavity 2
                N2_dBm = power_to_dBm(N2_watts)
                N2_dB = dBm_to_dB(N2_dBm)
                alpha2_solutions[i, j] = N2_dB

            except Exception as e:

                # print(f'Error at frequency {omega_d_val} and attenuation {attenuation}: {e}')
                alpha2_solutions[i, j] = -25  # Set a default or error value

    return alpha2_solutions

frequencies_nh = np.linspace(6.012e9, 6.047e9, nh_num_points)
frequencies = np.linspace(5.935e9, 6.135e9, hermitian_num_points)

attenuations = np.linspace(0.0, 14.5, num_attenuations)

net_gains = calculate_net_gain(np.array(attenuations))

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
    # Call this function for each phase
    store_numerical_traces(frequencies, attenuations, 'hermitian')
    store_numerical_traces(frequencies_nh, attenuations, 'nonhermitian')