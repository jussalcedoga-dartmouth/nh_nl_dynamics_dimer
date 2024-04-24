import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import os
from scipy.optimize import approx_fprime
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf
import pandas as pd
import json

folder_plots = 'plots'
os.makedirs(f'{folder_plots}', exist_ok=True)

### Helper functions to deal with experimental data...
def extract_threshold_crossing(df_exp, threshold=-40):
    crossing_points = []
    for phase in np.unique(df_exp['Phase [rad]']):
        phase_data = df_exp[df_exp['Phase [rad]'] == phase]
        gain_above_threshold = phase_data[phase_data['Amplitude [dBm]'] > threshold]['Gain [dB]']
        if not gain_above_threshold.empty:
            min_gain = gain_above_threshold.min()
            crossing_points.append((phase, min_gain))
    return np.array(crossing_points)

def load_csv(df_name):
    df_exp = pd.read_csv(df_name)
    ### Ignoring the single point where the transmission at the non-Hermitian phase goes significantly above 0 dB
    df_exp = df_exp[df_exp['Gain [dB]'] <= 9.5]
    return df_exp

df_no_delta_phi = load_csv('experimental_data/gain_phase_amplitude_no_deltaphi.csv')
df_delta_phi = load_csv('experimental_data/gain_phase_amplitude_deltaphi.csv')

threshold_crossing_points_no_delta_phi = extract_threshold_crossing(df_no_delta_phi)
threshold_crossing_points_delta_phi = extract_threshold_crossing(df_delta_phi)

## load the numerical parameters from the JSON
with open('numerical_parameters.json', 'r') as file:
    params = json.load(file)

## helper scaling params
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

kappa_cavity1 = kappa_0_1 + readout_kappa + t_hop_2_1 + t_hop_1_1 + extra_dissipation
kappa_cavity2 = kappa_0_2 + drive_kappa + t_hop_2_1 + t_hop_1_1 + extra_dissipation

epsilon_dBm = params['epsilon_dBm'] #dBm 
gain_threshold = params['gain_threshold'] #dB
G_0 = params['G_0']

max_symm_gain = 9.5 ### The max gain queried directly from the hashmap
insertion_loss = G_0 - max_symm_gain

grid_points = params['grid_phase_diagram']

gain_1 = G_0
a_1, b_1, c_1, flat_line_value_1, x0_1 = pwf.return_params(gain_1)

gain_2 = G_0
a_2, b_2, c_2, flat_line_value_2, x0_2 = pwf.return_params(gain_2)

def calculate_net_gain(attenuation, G_0=G_0, insertion_loss=insertion_loss):
    return G_0 - insertion_loss - attenuation

def dbm_to_watts(dbm):
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

def piece_wise_amp_function(x, a, b, c, flat_line_value, x0):
    return np.where(x <= x0, flat_line_value, model_function(x, a, b, c))

def dBm_to_dB(S21_dBm, input_power_dBm=-10):
    relative_power_dBm = S21_dBm
    relative_power_dB = relative_power_dBm - input_power_dBm
    return relative_power_dB

def func_real_no_drive(alpha, phase, attenuation):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * t_hop_1_1/4
    N2_watts = h_bar * omega2 * N2 * t_hop_1_2/4

    G1 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * 10 ** (-(attenuation + insertion_loss)/20)
    G2 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * 10 ** (-(attenuation + insertion_loss)/20)

    ## Equations of Motion
    d_alpha1 = -(((kappa_cavity1)/2 - G2*np.sqrt(t_hop_1_1 * t_hop_2_1)/2) + 1j * (omega1)) * alpha1_c + 1j * G2 * np.sqrt(t_hop_1_1 * t_hop_2_1)*alpha2_c
    d_alpha2 = -(((kappa_cavity2)/2 - G1*np.sqrt(t_hop_1_2 * t_hop_2_2)*np.exp(-1j * phase)/2) + 1j * (omega2)) * alpha2_c + 1j * G1 * np.sqrt(t_hop_1_2 * t_hop_2_2)*np.exp(-1j * phase)*alpha1_c 

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

def jacobian_numerical(alpha, phase, attenuation, epsilon=1e-12):
    func_to_diff = lambda x: func_real_no_drive(x, phase, attenuation)
    return approx_fprime(alpha, func_to_diff, epsilon)

def fixed_points(phase_val, attenuation, initial_guess):
    func_to_optimize = lambda x: func_real_no_drive(x, phase_val, attenuation)
    sol = root(func_to_optimize, initial_guess, 
               jac=lambda x: jacobian_numerical(x, phase_val, attenuation), tol=1e-20, method='hybr')
    
    if sol.success:
        # Calculate the Jacobian at the solution
        jacobian_at_sol = jacobian_numerical(sol.x, phase_val, attenuation)
        return sol.x, jacobian_at_sol
    else:
        return None, None

def calculate_power(N, kappa, hbar, omega):
    return N * kappa * hbar * omega / 4

# Function to convert power in Watts to dBm
def power_to_dBm(power_watts):
    return 10 * np.log10(power_watts * 1e3)

def check_stability(jacobian):
    eigenvalues = np.linalg.eigvals(jacobian)
    return np.all(np.real(eigenvalues) < 0)

def max_real_part_eigenvalues(jacobian):
    eigenvalues = np.linalg.eigvals(jacobian)
    return np.max(np.real(eigenvalues))

def create_stability_plot_driveless(net_gains):
    dBm_map = np.full((len(specific_phases), len(attenuation_values)), np.nan)

    for i, phase_val in enumerate(specific_phases):
        for j, att in enumerate(attenuation_values):
            global attenuation
            attenuation = att
            initial_guess = [1e6, 1e6, 1e6, 1e6]

            result, jacobian = fixed_points(phase_val, att, initial_guess)
            if result is not None:
                max_real_part = max_real_part_eigenvalues(jacobian)
                photon_number = max_real_part  
                power_watts = calculate_power(photon_number, readout_kappa, h_bar, omega1)
                dBm_value = power_to_dBm(power_watts)
                
                #### Let's just add a random quantity to make the scale match.
                ### This is not intended to refer to the amplitude of the limit cycle, tho
                dBm_map[i, j] = dBm_value + 80

    plt.figure(figsize=(10, 7))
    plt.xlabel(r'Net Gain [dB]', fontsize=20)
    plt.ylabel('$\phi$ [rads]', fontsize=20)
    plt.imshow(dBm_map, extent=[net_gains[0], net_gains[-1], specific_phases[-1], specific_phases[0]], aspect='auto', cmap='inferno', vmin=-45, vmax=0)
    plt.axvline(x=gain_threshold, ls = '--', lw=3.0, color='k')
    cbar = plt.colorbar()
    cbar.set_label(r'$\rm{max(Re}(\lambda))$', size=17)
    cbar.ax.tick_params(labelsize=20)
    plt.gca().invert_yaxis()

    if threshold_crossing_points_no_delta_phi.size > 0:
        plt.scatter(threshold_crossing_points_no_delta_phi[:, 1], threshold_crossing_points_no_delta_phi[:, 0], marker='o', s = 50,
                    color='crimson', edgecolors='crimson', label=r'No $\Delta \phi$')

    if threshold_crossing_points_delta_phi.size > 0:
        plt.scatter(threshold_crossing_points_delta_phi[:, 1], threshold_crossing_points_delta_phi[:, 0], marker='o', s = 50,
                    color='royalblue', edgecolors='royalblue', label=r'$\Delta \phi$')

    plt.legend(loc = 'upper left')
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(f'{folder_plots}/combined_stability_map.png', dpi=300)

# Define your attenuation ranges and specific phases
attenuation_values = np.linspace(0.0, 6.2, grid_points)[::-1]
specific_phases = np.linspace(0, 2*np.pi, grid_points)
net_gains = calculate_net_gain(np.array(attenuation_values))

create_stability_plot_driveless(net_gains)

print("Driveless dBm Map Analysis Completed!")