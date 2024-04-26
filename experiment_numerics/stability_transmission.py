import numpy as np
from scipy.optimize import root, fsolve, root_scalar, least_squares
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf
from scipy.optimize import approx_fprime
from matplotlib import cm
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

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

def func(alpha, omega_d, phase, attenuation):

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

def deriv_G12(N_12, mode_idx = 1):
    '''To be filled out by Juan. This function takes in the variable N1 (if mode_idx = 1),
       or N2 (if mode_idx = 2), and returns the derivative of the gain function G1 (if mode_idx = 1),
       or G2 (if mode_idx = 2) with respect to the variable N1 or N2 repectively
       -- so NO WATTS or anything, but taking in the actual (dimensionless) photon numbers...'''
    pass

def jacobian(alpha, omega_d, phase, attenuation, epsilon_dBm = None): # don't need drive for gradient...
    '''This function (written by Michiel) takes in the same input as the "func" method,
       but returns the Jacobian of the system of equations. It has two relations to other functions.
       First, the input "alpha" is typically taken as the output of the "fixed_points" function,
       since we want to evaluate the jacobian at an equilibrium point.
       Second, the method "" will further diagonalize the output of "jacobian",
       and decide on stability.'''
    
    # First part is same as for teh vector field defined in "func".
    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * t_hop_1_1/4
    N2_watts = h_bar * omega2 * N2 * t_hop_1_2/4

    G1 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * 10 ** (-(attenuation + insertion_loss)/20)
    G2 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * 10 ** (-(attenuation + insertion_loss)/20)

    jacobian = np.zeros((4, 4))

    # now fill out the jacobian with what we had found in the notebook "calculate_jacobian_analytically.ipynb"
    jacobian[0,0] = -kappa_cavity1/2 + t_hop_1_1*t_hop_2_1*G2/2
    jacobian[0,1] = omega1 - omega_d
    jacobian[0,2] = alpha1*alpha2*t_hop_1_1*t_hop_2_1 * deriv_G12(N2, mode_idx = 2)\
                    - 2*alpha2*alpha2_i*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1) * deriv_G12(N2, mode_idx = 2)
    jacobian[0,3] = -G2*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1)\
                    + alpha1*alpha2_i*deriv_G12(N2, mode_idx = 2)*t_hop_1_1*t_hop_2_1\
                    - 2*alpha2_i**2*deriv_G12(N2, mode_idx = 2)*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1)
    jacobian[1,0] = -omega1 + omega_d
    jacobian[1,1] = G2*t_hop_1_1*t_hop_2_1/2 - kappa_cavity1/2
    jacobian[1,2] = G2*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1)\
                    + alpha1_i*alpha2*deriv_G12(N2, mode_idx = 2)*t_hop_1_1*t_hop_2_1\
                    + 2*alpha2**2*deriv_G12(N2, mode_idx = 2)*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1)
    jacobian[1,3] = alpha1_i*alpha2_i*deriv_G12(N2, mode_idx = 2)*t_hop_1_1*t_hop_2_1\
                    + 2*alpha2*alpha2_i*deriv_G12(N2, mode_idx = 2)*np.sqrt(t_hop_1_1)*np.sqrt(t_hop_2_1)
    jacobian[2,0] = G1*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*np.sin(phase)\
                    + alpha1*alpha2*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.cos(phase)\
                    + alpha1*alpha2_i*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.sin(phase)\
                    - 2*alpha1*deriv_G12(N1, mode_idx = 1)*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*(-alpha1*np.sin(phase) + alpha1_i*np.cos(phase))
    jacobian[2,1] = -G1*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*np.cos(phase)\
                    + alpha1_i*alpha2*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.cos(phase)\
                    + alpha1_i*alpha2_i*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.sin(phase)\
                    - 2*alpha1_i*deriv_G12(N1, mode_idx = 1)*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*(-alpha1*np.sin(phase) + alpha1_i*np.cos(phase))
    jacobian[2,2] = G1*t_hop_1_2*t_hop_2_2*np.cos(phase)/2 - kappa_cavity2/2
    jacobian[2,3] = G1*t_hop_1_2*t_hop_2_2*np.sin(phase)/2 + omega2 - omega_d
    jacobian[3,0] = G1*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*np.cos(phase)\
                    - alpha1*alpha2*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.sin(phase)\
                    + alpha1*alpha2_i*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.cos(phase)\
                    + 2*alpha1*deriv_G12(N1, mode_idx = 1)*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*(alpha1*np.cos(phase) + alpha1_i*np.sin(phase))
    jacobian[3,1] = G1*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*np.sin(phase)\
                    - alpha1_i*alpha2*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.sin(phase)\
                    + alpha1_i*alpha2_i*deriv_G12(N1, mode_idx = 1)*t_hop_1_2*t_hop_2_2*np.cos(phase)\
                    + 2*alpha1_i*deriv_G12(N1, mode_idx = 1)*np.sqrt(t_hop_1_2)*np.sqrt(t_hop_2_2)*(alpha1*np.cos(phase) + alpha1_i*np.sin(phase))
    jacobian[3,2] = -G1*t_hop_1_2*t_hop_2_2*np.sin(phase)/2 - omega2 + omega_d
    jacobian[3,3] = G1*t_hop_1_2*t_hop_2_2*np.cos(phase)/2 - kappa_cavity2/2

    return jacobian

def calculate_power(N, kappa, hbar, omega):
    return N * kappa * hbar * omega/4

# Function to convert power in Watts to dBm
def power_to_dBm(power_watts):
    return 10 * np.log10(power_watts * 1e3)

def check_stability(jacobian):
    '''This looks good! We should be all set with the new
       analytical way of determining the jacobian (Michiel).'''
    eigenvalues = np.linalg.eigvals(jacobian)
    return np.all(np.real(eigenvalues) < 0)

def jacobian_numerical(alpha, omega_d, phase, att, epsilon=1e-8):
    """ Calculate the numerical Jacobian. """
    func_to_diff = lambda x: func(x, omega_d, phase, att)
    return approx_fprime(alpha, func_to_diff, epsilon)

def fixed_points_jac(omega_d_val, phase_val, att, initial_guess):
    """ Attempt to find the fixed points and compute the Jacobian at those points. """
    func_to_optimize = lambda x: func(x, omega_d_val, phase_val, att)
    sol = root(func_to_optimize, initial_guess, method='hybr')
    
    if sol.success:
        # Calculate the Jacobian at the solution
        jacobian_at_sol = jacobian_numerical(sol.x, omega_d_val, phase_val, att)
        return sol.x, jacobian_at_sol
    else:
        return None, None

def max_real_part_eigenvalues(jacobian):
    eigenvalues = np.linalg.eigvals(jacobian)
    return np.max(np.real(eigenvalues))

def get_stability_transmission(frequency_values, attenuation_values, phase, alias_phase):
    net_gains = calculate_net_gain(np.array(attenuation_values))

    # Create mesh grids for frequency and attenuation
    frequency_grid, attenuation_grid = np.meshgrid(frequency_values, attenuation_values)

    # Initialize an array to hold the stability information
    stability_array = np.zeros_like(frequency_grid, dtype=float)

    # Loop over all frequencies and attenuations
    for i, freq in enumerate(frequency_values):
        for j, att in enumerate(attenuation_values):
            initial_guess = [1e8, 1e8, 1e8, 1e8]  # Adjusted for a more realistic initial guess
            result, jacobian = fixed_points_jac(freq, phase, att, initial_guess)
            if result is not None and jacobian is not None:
                # Check the eigenvalues for stability analysis
                if check_stability(jacobian):
                    stability_array[j, i] = 0  # Stable
                else:
                    # Record the maximum real part of the eigenvalues if unstable
                    stability_array[j, i] = max_real_part_eigenvalues(jacobian)

    # Plotting the stability as a 2D colorplot
    plt.figure(figsize=(4.5, 5))
    c = plt.pcolormesh(frequency_grid / 1e9, net_gains, stability_array, cmap='coolwarm', shading='auto')
    plt.colorbar(c, label='Max Real Part of Eigenvalues')
    plt.axhline(y=gain_threshold, color='k', linestyle='--', label='Threshold')  # Threshold line
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Attenuation [dB]')
    plt.title(f'Stability Analysis - {alias_phase.capitalize()} Phase')
    plt.tight_layout()
    plt.savefig(f'images_phase_{alias_phase}/stability_analysis_{alias_phase}.png')
    plt.close()

# Example usage
attenuation_values = np.linspace(0, 14.5, num_attenuations)  # num_attenuations needs to be defined

# hermitian
frequency_values = np.linspace(5.935e9, 6.135e9, hermitian_num_points)  # hermitian_num_points needs to be defined
get_stability_transmission(frequency_values, attenuation_values, 0.0, 'hermitian')

# non-Hermitian
frequencies_nh = np.linspace(6.012e9, 6.047e9, nh_num_points)  # nh_num_points needs to be defined
get_stability_transmission(frequencies_nh, attenuation_values, np.pi, 'nonhermitian')

print('Stability Analysis for transmission sweep completed!!')