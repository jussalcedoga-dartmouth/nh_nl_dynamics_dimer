import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import os
from scipy.optimize import approx_fprime
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

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

# threshold_crossing_points_no_delta_phi = extract_threshold_crossing(df_no_delta_phi)
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

max_symm_gain = 9.5 ### The max gain queried directly from the hashmap
insertion_loss = G_0 - max_symm_gain

# Function to calculate net gain from attenuation
def calculate_net_gain(attenuation, G_0=G_0, insertion_loss=insertion_loss):
    return G_0 - insertion_loss - attenuation

grid_points = params['grid_phase_diagram']

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

def model_function_reflection(x, a, b, c):
    return -a * (1 / (1 + (x / b))) + c

## fitted values from transmission and reflection from the amp

## transmission
gain_1 = G_0
a_1, b_1, c_1, flat_line_value_1, x0_1 =  9.2448, 0.008638, 0.0, 8.2716171, 0.0007981

## transmission
gain_2 = G_0
a_2, b_2, c_2, flat_line_value_2, x0_2 =  9.2448, 0.008638, 0.0, 8.2716171, 0.0007981

## reflections...
a_r, b_r, c_r, flat_line_r, x0_r = 0.3554, 0.00634, 0.73626, 0.4854, 0.0007981

def piece_wise_amp_function(x, a, b, c, flat_line_value, x0):
    return np.where(x <= x0 + 0.0002, flat_line_value, model_function(x, a, b, c))

def piece_wise_amp_reflection(x_r, a_r, b_r, c_r, flat_line_r, x0_r):
    return np.where(x_r <= x0_r + 0.002, flat_line_r, model_function_reflection(x_r, a_r, b_r, c_r))

def model_function_reflection_ps_da(x, slope, intercept):
    return slope * x + intercept

def dBm_to_dB(S21_dBm, input_power_dBm=-10):
    relative_power_dBm = S21_dBm
    relative_power_dB = relative_power_dBm - input_power_dBm
    return relative_power_dB

def find_closest_parameters(df, phase_value):
    # Find the index of the row with the closest phase value
    closest_idx = (df['phase'] - phase_value).abs().idxmin()
    # Find the index of the row with the closest new phase value    
    # Extract the parameters for the closest new phase
    closest_params = df.loc[closest_idx, ['slope', 'intercept']]
    slope = closest_params['slope']
    intercept = closest_params['intercept']

    return slope, intercept

def find_closest_gain_parameters(df, gain_value):
    # Find the index of the row with the closest gain value
    closest_idx = (df['gain'] - gain_value).abs().idxmin()
    
    # Extract the parameters for the closest gain
    closest_params = df.loc[closest_idx, ['slope', 'intercept']]
    slope = closest_params['slope']
    intercept = closest_params['intercept']

    return slope, intercept

# Load the CSV file once outside the function
csv_path = 'csvs_calibrations/linear_fit_parameters_by_phase.csv'
df = pd.read_csv(csv_path)

csv_path_da = 'csvs_calibrations/linear_fit_parameters_by_gain.csv'
df_da = pd.read_csv(csv_path_da)

from collections import defaultdict

kappa_ts_dict = defaultdict(list)
kappa_ts2_dict = defaultdict(list)

Jeff_12_dict = defaultdict(list)
Jeff_21_dict = defaultdict(list)
N1_dict = defaultdict(list)
N2_dict = defaultdict(list)

def func_real_no_drive(alpha, phase, attenuation):

    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * kappa_c/4
    N2_watts = h_bar * omega2 * N2 * kappa_c/4

    eta_A = 10**(-attenuation/20)
    eta_I = 10**(-insertion_loss/20)

    ## Gain
    nu_G12 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * eta_A * eta_I
    nu_G21 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * eta_A * eta_I

    ## Reflections off the amp
    nu_r1 = piece_wise_amp_reflection(N1_watts, a_r, b_r, c_r, flat_line_r, x0_r)
    nu_r2 = piece_wise_amp_reflection(N2_watts, a_r, b_r, c_r, flat_line_r, x0_r)

    ## Reflections off the phase shifter
    slope_ps, intercept_ps = find_closest_parameters(df, phase)
    nu_ps = model_function_reflection_ps_da(N2_watts, slope_ps, intercept_ps)*eta_A

    ## Reflections off the attenuator
    query_gain_value = max_symm_gain - attenuation
    slope_da, intercept_da = find_closest_gain_parameters(df_da, query_gain_value)
    nu_da = model_function_reflection_ps_da(N1_watts, slope_da, intercept_da)

    if nu_G12 >= 1.0:
        kappa_diag_1 = kappa_0_1 + drive_kappa + kappa_c - (nu_r1 + nu_ps)*kappa_c - (nu_G21 - 1)*kappa_c
        kappa_diag_2 = kappa_0_2 + readout_kappa + kappa_c - (nu_r2 + nu_da)*kappa_c - (nu_G12 - 1)*kappa_c
    else:
        kappa_diag_1 = kappa_0_1 + drive_kappa + kappa_c - (nu_r1 + nu_ps)*kappa_c
        kappa_diag_2 = kappa_0_2 + readout_kappa + kappa_c - (nu_r2 + nu_da)*kappa_c

    print('Photon Numbers:', N1_watts, N2_watts)
    print('\n', kappa_diag_1/1e6, kappa_diag_2/1e6, ', gain: ', query_gain_value, ', nu_da: ', nu_da, ', nu_r1: ', nu_r1, ', nu_r2: ', nu_r2, ', nu_ps: ', nu_ps)

    kappa_ts_dict[(attenuation, phase)].append(kappa_diag_1)
    kappa_ts2_dict[(attenuation, phase)].append(kappa_diag_2)

    d_alpha1 = -(kappa_diag_1 + 1j*(omega1))*alpha1_c \
                - 1j * nu_G21*np.exp(1j*(phase))*kappa_c*alpha2_c

    d_alpha2 = -(kappa_diag_2 + 1j*(omega2))*alpha2_c \
                - 1j * nu_G12*kappa_c*alpha1_c
    
    Jeff_12_dict[(attenuation, phase)].append(nu_G12*kappa_c)
    Jeff_21_dict[(attenuation, phase)].append(nu_G21*kappa_c)

    Jeff_21_dict[(attenuation, phase)].append(nu_G21*kappa_c)
    N1_dict[(attenuation, phase)].append(N1_watts)
    N2_dict[(attenuation, phase)].append(N2_watts)
    #################################################################################################

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
    plt.imshow(dBm_map, extent=[net_gains[0], net_gains[-1], specific_phases[-1], specific_phases[0]], aspect='auto', cmap='inferno', vmin=-45)#, vmax=0)
    plt.axvline(x=gain_threshold, ls = '--', lw=3.0, color='k')
    cbar = plt.colorbar()
    cbar.set_label(r'$\rm{max(Re}(\lambda))$', size=17)
    cbar.ax.tick_params(labelsize=20)
    plt.gca().invert_yaxis()

    # if threshold_crossing_points_no_delta_phi.size > 0:
    #     plt.scatter(threshold_crossing_points_no_delta_phi[:, 1], threshold_crossing_points_no_delta_phi[:, 0], marker='o', s = 50,
    #                 color='crimson', edgecolors='crimson', label=r'No $\Delta \phi$')

    if threshold_crossing_points_delta_phi.size > 0:
        plt.scatter(threshold_crossing_points_delta_phi[:, 1], threshold_crossing_points_delta_phi[:, 0], marker='o', s = 50,
                    color='crimson', edgecolors='crimson', label=r'$\Delta \phi$')

    plt.legend(loc = 'upper left')
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(f'{folder_plots}/combined_stability_map.png', dpi=300)

def set_yaxis_ticks(ax, tick_size=23):
    # Set major locator to MultipleLocator (pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
    # Define custom formatter to handle specific label formatting
    def format_func(value, pos):
        if np.isclose(value, 0):
            return r'0'
        elif np.isclose(value, np.pi):
            return r'$\pi$'  # Display 'π' instead of '1π'
        elif np.isclose(value, 2 * np.pi):
            return r'2$\pi$'  # Optionally handle the 2π case as well
        else:
            return f'{value/np.pi:.1g}$\pi$'  # General case

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # Increase tick size
    ax.tick_params(axis='y', which='major', labelsize=tick_size)

# Define your attenuation ranges and specific phases
# attenuation_values = np.linspace(0.0, 6.2, grid_points)[::-1]
attenuation_values = np.linspace(0.0, 14.5, grid_points)[::-1]
specific_phases = np.linspace(0, 2*np.pi, grid_points)
net_gains = calculate_net_gain(np.array(attenuation_values))

create_stability_plot_driveless(net_gains)

print("Driveless dBm Map Analysis Completed!")

#### On-diagonals
#################################################################################################
### Let's just plot the numbers
# Compute the average kappa for each (attenuation, phase)
attenuations = sorted(set(k for k, p in kappa_ts_dict.keys()))
phases = sorted(set(p for k, p in kappa_ts_dict.keys()))

# Calculate net gains as inverse of attenuations (assumed linear relationship for simplicity)
net_gains = sorted(max(attenuations) - att for att in attenuations)

# Prepare the data grids
kappa_avg_grid = np.zeros((len(phases), len(net_gains)))
kappa_T_1_grid = np.zeros((len(phases), len(net_gains)))
kappa_T_2_grid = np.zeros((len(phases), len(net_gains)))

for i, phase in enumerate(phases):
    for j, attenuation in enumerate(attenuations):
        net_gain_index = net_gains.index(max(attenuations) - attenuation)
        kappa1_avg = np.mean(kappa_ts_dict[(attenuation, phase)])
        kappa2_avg = np.mean(kappa_ts2_dict[(attenuation, phase)])

        kappa_avg_grid[i, net_gain_index] = np.mean([kappa1_avg, kappa2_avg])/1e6  # Convert to MHz
        kappa_T_1_grid[i, net_gain_index] = kappa1_avg / 1e6  # Convert to MHz
        kappa_T_2_grid[i, net_gain_index] = kappa2_avg / 1e6  # Convert to MHz

net_gains = calculate_net_gain(np.array(attenuation_values))

#################################################################################################
# Plotting Average Kappa
fig, ax = plt.subplots()
c = ax.imshow(kappa_avg_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$\tilde{\kappa}_{T}$ [MHz]')
plt.tight_layout()
plt.savefig(f'{folder_plots}/kappas_avg_colorplot.png')
plt.close()

# Plotting Kappa T_1
fig, ax = plt.subplots()
c = ax.imshow(kappa_T_1_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$\tilde{\kappa}_{T_{1}}$ [MHz]')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/kappas_1_colorplot.png')
plt.close()

# Plotting Kappa T_2
fig, ax = plt.subplots()
c = ax.imshow(kappa_T_2_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$\tilde{\kappa}_{T_{2}}$ [MHz]')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/kappas_2_colorplot.png')
plt.close()
#################################################################################################

### Off-diagonals
#################################################################################################
attenuations = sorted(set(k for k, p in Jeff_12_dict.keys()))
phases = sorted(set(p for k, p in Jeff_12_dict.keys()))

# Assuming net gain inversely related to attenuation (net_gain = max(attenuations) - attenuation)
net_gains = sorted(max(attenuations) - att for att in attenuations)

# Prepare the data grids
Jeff_12_grid = np.zeros((len(phases), len(net_gains)))
Jeff_21_grid = np.zeros((len(phases), len(net_gains)))
Jeff_avg_grid = np.zeros((len(phases), len(net_gains)))

for i, phase in enumerate(phases):
    for j, attenuation in enumerate(attenuations):
        net_gain_index = net_gains.index(max(attenuations) - attenuation)
        Jeff_12_avg = np.mean(Jeff_12_dict[(attenuation, phase)])
        Jeff_21_avg = np.mean(Jeff_21_dict[(attenuation, phase)])

        Jeff_avg_grid[i, net_gain_index] = np.mean([Jeff_12_avg, Jeff_21_avg]) / 1e6
        Jeff_12_grid[i, net_gain_index] = Jeff_12_avg / 1e6
        Jeff_21_grid[i, net_gain_index] = Jeff_21_avg / 1e6

net_gains = calculate_net_gain(np.array(attenuation_values))

# Plotting Average Jeff
fig, ax = plt.subplots()
c = ax.imshow(Jeff_avg_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$\tilde{J}$ [MHz]')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/Jeff_avg_colorplot.png')
plt.close()

# Plotting Jeff_12
fig, ax = plt.subplots()
c = ax.imshow(Jeff_12_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$\tilde{J}_{12}$ [MHz]')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/Jeff_12_colorplot.png')
plt.close()

# Plotting Jeff_21
fig, ax = plt.subplots()
c = ax.imshow(Jeff_21_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
cbar.set_label(r'$\tilde{J}_{21}$ [MHz]')
plt.tight_layout()
plt.savefig(f'{folder_plots}/Jeff_21_colorplot.png')
plt.close()

#### Single trace for J as a function of Delta G
phase_index = 0  # you can change this index to select a different phase
selected_phase = phases[phase_index]

# Extract the trace for the selected phase
J_avg_trace = Jeff_avg_grid[phase_index, :]
net_gains = calculate_net_gain(np.array(attenuation_values))

# Plotting J as a function of net gain for the selected phase
fig, ax = plt.subplots()
ax.plot(net_gains, J_avg_trace, ls='-', lw=3.0, color='crimson')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\tilde{J}$ [MHz]')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/J_vs_deltaG.png')
plt.close()

#### Single trace for J as a function of Delta G
phase_index = 0  # you can change this index to select a different phase
selected_phase = phases[phase_index]

# Extract the trace for the selected phase
J_avg_trace = Jeff_avg_grid[phase_index, :]
net_gains = calculate_net_gain(np.array(attenuation_values))

# Plotting J as a function of net gain for the selected phase
fig, ax = plt.subplots()
ax.plot(net_gains, J_avg_trace/(kappa_c/1e6), ls='-', lw=3.0, color='crimson')
ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\tilde{J}/\kappa_c$')

plt.axhline(1.0, ls='--', color='k')
plt.axvline(4.52, ls='--', color='k')

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/Jk_c_vs_deltaG.png')
plt.close()

net_gains = sorted(max(attenuations) - att for att in attenuations)

# Prepare the data grids
N1_grid = np.zeros((len(phases), len(net_gains)))
N2_grid = np.zeros((len(phases), len(net_gains)))
N_avg_grid = np.zeros((len(phases), len(net_gains)))

for i, phase in enumerate(phases):
    for j, attenuation in enumerate(attenuations):
        net_gain_index = net_gains.index(max(attenuations) - attenuation)
        N1_avg = np.mean(N1_dict[(attenuation, phase)])
        N2_avg = np.mean(N2_dict[(attenuation, phase)])
        N_avg = np.mean([N1_avg, N2_avg])

        N1_grid[i, net_gain_index] = N1_avg
        N2_grid[i, net_gain_index] = N2_avg
        N_avg_grid[i, net_gain_index] = N_avg

net_gains = calculate_net_gain(np.array(attenuation_values))

# Plotting Average N1
fig, ax = plt.subplots()
c = ax.imshow(N1_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')

ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$N_1$')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/N1_colormap.png')
plt.close()

# Plotting Average N2
fig, ax = plt.subplots()
c = ax.imshow(N2_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')

ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$N_2$')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/N2_colormap.png')
plt.close()

# Plotting Average N2
fig, ax = plt.subplots()
c = ax.imshow(N_avg_grid, aspect='auto', origin='lower', 
              extent=[net_gains[0], net_gains[-1], phases[0], phases[-1]],
              interpolation='nearest', cmap='inferno')

ax.set_xlabel(r'$\Delta G$ [dB]')
ax.set_ylabel(r'$\phi$')
set_yaxis_ticks(ax)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$N_avg$')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig(f'{folder_plots}/N_avg_colormap.png')
plt.close()