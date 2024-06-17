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

## For individual numerical plots... I am mostly interested in them compared to experiments.
save_individual = False

## create folder to save the plots
folder_name = 'plots'
os.makedirs(folder_name, exist_ok=True)

## load the numerical parameters from the JSON
with open('numerical_parameters_crossover.json', 'r') as file:
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

## fitted values from transmission and reflection from the amp
gain_1 = G_0
a_1, b_1, c_1, flat_line_value_1, x0_1 =  9.177, 0.00839, 0.0930, 8.2716, 0.000798

gain_2 = G_0
a_2, b_2, c_2, flat_line_value_2, x0_2 =  9.177, 0.00839, 0.0930, 8.2716, 0.000798

## reflections...
a_r, b_r, c_r, flat_line_r, x0_r = 0.3554, 0.00634, 0.73626, 0.4854, 0.0007981

def piece_wise_amp_function(x, a, b, c, flat_line_value, x0):
    return np.where(x <= x0, flat_line_value, model_function(x, a, b, c))

def piece_wise_amp_reflection(x_r, a_r, b_r, c_r, flat_line_r, x0_r):
    return np.where(x_r <= x0_r, flat_line_r, model_function_reflection(x_r, a_r, b_r, c_r))

def model_function_reflection_ps(x, a, b, c):
    return -a * (1 / (1 + (x / b))) + c

def dBm_to_dB(S21_dBm, input_power_dBm=-10):
    relative_power_dBm = S21_dBm
    relative_power_dB = relative_power_dBm - input_power_dBm
    return relative_power_dB

kappa_ts_dict = defaultdict(list)

def func(alpha, omega_d, phase, attenuation, epsilon_dBm = -10):

    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i
    
    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((drive_kappa * epsilon_watts) / (h_bar * omega_d))
    
    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2
    
    if phase == 0:
        crossover = 10**(0.3/20)
        kappa_c = 18.8e6
    else:
        crossover = 10**(1.5/20)
        kappa_c = 16.8e6

    N1_watts = h_bar * omega1 * N1 * kappa_c/4
    N2_watts = h_bar * omega2 * N2 * kappa_c/4

    eta_A = 10**(-attenuation/20)
    eta_I = 10**(-insertion_loss/20)

    nu_G12 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * eta_A * eta_I
    nu_G21 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * eta_A * eta_I

    nu_r1 = piece_wise_amp_reflection(N1_watts, a_r, b_r, c_r, flat_line_r, x0_r) * eta_A * eta_I
    nu_r2 = piece_wise_amp_reflection(N2_watts, a_r, b_r, c_r, flat_line_r, x0_r) * eta_A * eta_I

    ### Reflections off the phase shifter. Fitted from the actual data at different phases
    if phase == 0:
        ## Values from the fit at phase = 0
        a_ps, b_ps, c_ps = 1.5935288315890137, 0.003331854666549762, 1.6466953192039548

    else:
        ## Values from the fit at a phase close to pi
        a_ps, b_ps, c_ps = 0.6272190505200504, 0.0010786013218081418, 0.6621421355283915

    nu_ps = model_function_reflection_ps(N1_watts, a_ps, b_ps, c_ps)

    # take into account phase dependent reflections that modify the loss rate
    kappa_diag_1 = crossover * kappa_c 
    kappa_diag_2 = crossover * kappa_c

    J_12 = nu_G12 * kappa_c
    J_21 = nu_G21 * kappa_c

    # including saturation effects
    delta_crossover = (nu_G12 - crossover)
    
    if nu_G12 > crossover:
        kappa_diag_1 -= delta_crossover * kappa_c
        kappa_diag_2 -= delta_crossover * kappa_c
        J_21 -= nu_r1 * delta_crossover * kappa_c
        J_12 -= nu_ps * nu_r2 * delta_crossover * kappa_c 

    kappa_ts_dict[attenuation].append(kappa_diag_1)

    return attenuation, kappa_diag_1

def fixed_points(omega_d_val, phase_val, att, initial_guess, epsilon_dBm=-10):

    if phase_val == 0:
        tolerance = 1e-8
        sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='hybr')
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

        for i, omega_d_val in enumerate(frequencies):
            try:
                # Use last successful guess as the initial condition
                alpha1_sol, _ = fixed_points(omega_d_val, phase, attenuation, initial_guess, epsilon_dBm=epsilon_dBm)

                ## Compute photon numbers and convert it to a dB scale to compare with experiment
                N1_total = np.sqrt(alpha1_sol.real**2 + alpha1_sol.imag**2)**2
                N2_watts = calculate_power(N1_total, readout_kappa, h_bar, omega1)  # Readout cavity 1
                N2_dBm = power_to_dBm(N2_watts)
                N2_dB = dBm_to_dB(N2_dBm)
                alpha2_solutions[i, j] = N2_dB

            except Exception as e:
                # print(f'Error at frequency {omega_d_val} and attenuation {attenuation}: {e}')
                alpha2_solutions[i, j] = -25 # Set a default or error value

    return alpha2_solutions

##############################################################################################################
### Hermitian results
frequencies = np.linspace(5.935e9, 6.135e9, hermitian_num_points)
attenuations = np.linspace(0.0, 14.5, num_attenuations)
phase = 0
alpha2_solutions = create_plots_for_phase(frequencies, attenuations, phase, epsilon_dBm=epsilon_dBm)
center_frequency = np.mean([omega1, omega2])/1e9
frequency_array = frequencies / 1e9  # Convert to GHz for plotting
net_gains = calculate_net_gain(np.array(attenuations))

kappa_ts_to_plot = []
plt.figure()
for k in kappa_ts_dict.keys():
    kappa_t = list(set(kappa_ts_dict[k]))[0]
    kappa_ts_to_plot.append(kappa_t/1e6)

plt.figure(figsize=(8, 5))
plt.plot(net_gains, kappa_ts_to_plot, '--', lw=2.0, label='$\kappa_T$')
plt.plot(net_gains, 2*np.array(kappa_ts_to_plot), '--', lw=2.0, label='2$\kappa_T$')
plt.xlabel('$\Delta G$', fontsize=20)
plt.ylabel('$\kappa_T$ [MHz]', fontsize=20)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('individual_k_ts/kappa_t_hermitian_phase.png')
plt.close()
##############################################################################################################


##############################################################################################################
### nonHermitian results
frequencies = np.linspace(5.935e9, 6.135e9, hermitian_num_points)
attenuations = np.linspace(0.0, 14.5, num_attenuations)
phase = np.pi
alpha2_solutions = create_plots_for_phase(frequencies, attenuations, phase, epsilon_dBm=epsilon_dBm)

center_frequency = np.mean([omega1, omega2])/1e9
frequency_array = frequencies / 1e9  # Convert to GHz for plotting
net_gains = calculate_net_gain(np.array(attenuations))

kappa_ts_to_plot = []
plt.figure()
for k in kappa_ts_dict.keys():
    kappa_t = list(set(kappa_ts_dict[k]))[0]
    kappa_ts_to_plot.append(kappa_t/1e6)

plt.figure(figsize=(8, 5))
plt.plot(net_gains, kappa_ts_to_plot, '--', lw=2.0, label='$\kappa_T$')
plt.plot(net_gains, 2*np.array(kappa_ts_to_plot), '--', lw=2.0, label='2$\kappa_T$')
plt.xlabel('$\Delta G$', fontsize=20)
plt.ylabel('$\kappa_T$', fontsize=20)

plt.legend(loc='best')
plt.tight_layout()
plt.savefig('individual_k_ts/kappa_t_nonhermitian_phase.png')
plt.close()
##############################################################################################################
