import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import os
from scipy.optimize import approx_fprime
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf

os.makedirs('phase_diagram_w_drive', exist_ok=True)

omega1 = 6.02982e9
omega2 = 6.029494e9

h_bar = 1.054571817e-34

t_hop_1_1 = 15e6#.71e6## sigmax cavity 1

t_hop_1_2 = 15e6#.89e6## port 1 cavity 2

t_hop_2_1 = 15e6#.77e6## sigmaz cavity 1

t_hop_2_2 = 15e6#.94e6## sigmax cavity 2

# Drive 2 read 1
readout_kappa = 3.06e6# port 1 cavity 1
drive_kappa = 3.87e6# port sigmaz cavity 2

kappa_0_1 = 7.04e6
kappa_0_2 = 8.73e6

h_bar = 1.054571817e-34

kappa_cavity1 = kappa_0_1 + readout_kappa + t_hop_2_1 + t_hop_1_1
kappa_cavity2 = kappa_0_2 + drive_kappa + t_hop_2_2 + t_hop_1_2

epsilon_dBm = -10
epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)

gain_1 = 19.7
a_1, b_1, c_1, flat_line_value_1, x0_1 = pwf.return_params(gain_1)

gain_2 = 19.7
a_2, b_2, c_2, flat_line_value_2, x0_2 = pwf.return_params(gain_2)

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

epsilon_dBm = 8

def func_real_no_drive(alpha, phase):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha
    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)

    epsilon = np.sqrt((drive_kappa * epsilon_watts) / (h_bar * omega2))

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    N1_watts = h_bar * omega1 * N1 * t_hop_1_1/4
    N2_watts = h_bar * omega2 * N2 * t_hop_1_2/4

    G1 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * 10 ** (-(attenuation + 10.4 + 0.9)/20)
    G2 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * 10 ** (-(attenuation + 6.7 + 4.5)/20)

    d_alpha1 = -(((kappa_cavity1) - (G2)*np.sqrt(t_hop_1_1 * t_hop_2_1)) + 1j * (omega1 - omega1)) * alpha1_c + 1j * G2 * np.sqrt(t_hop_1_1 * t_hop_2_1) * alpha2_c
    d_alpha2 = -(((kappa_cavity2) - (G1)*np.sqrt(t_hop_1_2 * t_hop_2_2)*np.exp(-1j * phase)) + 1j * (omega2 - omega1)) * alpha2_c + 1j * G1 * np.sqrt(t_hop_1_2 * t_hop_2_2) * alpha1_c * np.exp(-1j * phase) + epsilon

    return np.array([d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag])

def jacobian_numerical(alpha, phase, epsilon=1e-4):
    func_to_diff = lambda x: func_real_no_drive(x, phase)
    return approx_fprime(alpha, func_to_diff, epsilon)

def fixed_points(phase_val, initial_guess):
    func_to_optimize = lambda x: func_real_no_drive(x, phase_val)
    sol = root(func_to_optimize, initial_guess, 
               jac=lambda x: jacobian_numerical(x, phase_val), tol=1e-4, method='hybr')
    
    if sol.success:
        # Calculate the Jacobian at the solution
        jacobian_at_sol = jacobian_numerical(sol.x, phase_val)
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

def create_stability_plot_driveless():
    dBm_map = np.full((len(specific_phases), len(attenuation_values)), np.nan)

    for i, phase_val in enumerate(specific_phases):
        for j, att in enumerate(attenuation_values):
            global attenuation
            attenuation = att
            initial_guess = [1e6, 1e6, 1e6, 1e6]

            result, jacobian = fixed_points(phase_val, initial_guess)
            if result is not None:
                max_real_part = max_real_part_eigenvalues(jacobian)
                photon_number = max_real_part  
                power_watts = calculate_power(photon_number, readout_kappa, h_bar, omega1)
                dBm_value = power_to_dBm(power_watts)
                dBm_map[i, j] = dBm_value

    # Plotting the dBm map
    plt.figure(figsize=(7, 4))
    plt.imshow(dBm_map, extent=[attenuation_values[0], attenuation_values[-1], specific_phases[-1], specific_phases[0]], aspect='auto', cmap='inferno')
    plt.colorbar(label='dBm')
    plt.xlabel('Attenuation [dB]')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.savefig('phase_diagram_no_drive/dBm_map_driveless.png', dpi=300)
    plt.close()

# Define your attenuation ranges and specific phases
attenuation_values = np.linspace(0, 4.0, 50)[::-1]
specific_phases = np.linspace(0, 2*np.pi, 50)

create_stability_plot_driveless()

print("Driveless dBm Map Analysis Completed!")
