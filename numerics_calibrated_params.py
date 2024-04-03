import numpy as np
from scipy.optimize import root, fsolve, root_scalar, least_squares
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import saturation_curve_from_points as sim
import piece_wise_function_flat as pwf
from matplotlib import cm

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

folder_name = '230318_JS_calibrated_params_new_devices2'
os.makedirs(folder_name, exist_ok=True)

# Constants and Parameters
omega1 = 6.03582e9
omega2 = 6.03494e9

h_bar = 1.054571817e-34

t_hop_1_1 = 18e6#.71e6## sigmax cavity 1

t_hop_1_2 = 18e6#.89e6## port 1 cavity 2

t_hop_2_1 = 18e6#.77e6## sigmaz cavity 1

t_hop_2_2 = 18e6#.94e6## sigmax cavity 2

# Drive 2 read 1
readout_kappa = 3.06e6# port 1 cavity 1
drive_kappa = 3.87e6# port sigmaz cavity 2

kappa_0_1 = 7.04e6
kappa_0_2 = 8.73e6

# kappa_0_1 = omega1/(2*850)
# kappa_0_2 = omega2/(2*850)

kappa_cavity1 = kappa_0_1 + drive_kappa + t_hop_2_1 + t_hop_1_1
kappa_cavity2 = kappa_0_2 + readout_kappa + t_hop_2_2 + t_hop_1_2

epsilon_dBm = -10

## offsets at particular phases...
# offsets = {0.0: 4.5, np.pi: 5.2 + 8}
# vmaxs = {0.0: -16.54, np.pi: 5}

def dbm_to_watts(dbm):
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

gain_1 = 19.5
a_1, b_1, c_1, flat_line_value_1, x0_1 = pwf.return_params(gain_1)

gain_2 = 19.5
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

    G1 = piece_wise_amp_function(N1_watts, a_1, b_1, c_1, flat_line_value_1, x0_1) * 10 ** (-(attenuation + 10.4 + 0.9)/20)
    G2 = piece_wise_amp_function(N2_watts, a_2, b_2, c_2, flat_line_value_2, x0_2) * 10 ** (-(attenuation + 6.7 + 4.5)/20)

    d_alpha1 = -(((kappa_cavity1) - (G2)*np.sqrt(t_hop_1_1 * t_hop_2_1)) + 1j * (omega1 - omega_d)) * alpha1_c + 1j * G2 * np.sqrt(t_hop_1_1 * t_hop_2_1) * alpha2_c
    d_alpha2 = -(((kappa_cavity1) - (G1)*np.sqrt(t_hop_1_2 * t_hop_2_2)) + 1j * (omega2 - omega_d)) * alpha2_c + 1j * G1 * np.sqrt(t_hop_1_2 * t_hop_2_2) * alpha1_c * np.exp(-1j * phase) + epsilon

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

def fixed_points(omega_d_val, phase_val, att, initial_guess, epsilon_dBm=-10):

    if phase_val == 0:
        tolerance = 1e-8
    else:
        tolerance = 1e-4

    sol = root(lambda x: func(x, omega_d_val, phase_val, att, epsilon_dBm=epsilon_dBm), initial_guess, tol=tolerance, method='broyden1')
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

    max_values = []
    max_values_below_center = []
    max_values_above_center = []
    
    if phase == 0.0:
        initial_guess = [1e6, 1e6, 1e6, 1e6]
    else:
        initial_guess = [1e8, 1e8, 1e8, 1e8]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust to use inferno colormap
    colors = cm.inferno(np.linspace(0, 1, len(attenuations)))

    for j, attenuation in enumerate(attenuations):
        max_alpha_2_real = 0
        max_alpha_2_imag = 0

        transmission_below_center = []
        transmission_above_center = []

        for i, omega_d_val in enumerate(frequencies):
            try:
                alpha1_sol, alpha2_sol = fixed_points(omega_d_val, phase, attenuation, initial_guess, epsilon_dBm=epsilon_dBm)

                magnitude_2 = np.sqrt(alpha2_sol.real**2 + alpha2_sol.imag**2)**2
                magnitude_1 = np.sqrt(alpha1_sol.real**2 + alpha1_sol.imag**2)**2

                N2_total = magnitude_2
                N1_total = magnitude_1

                if alpha2_sol.real > max_alpha_2_real:
                    max_alpha_2_real = alpha2_sol.real
                if alpha2_sol.imag > max_alpha_2_imag:
                    max_alpha_2_imag = alpha2_sol.imag

                N2_watts = calculate_power(N1_total, readout_kappa, h_bar, omega2)

                N2_dBm = power_to_dBm(N2_watts)
                N2_dB = dBm_to_dB(N2_dBm)

                alpha2_solutions[i, j] = N2_dB

                center_frequency = np.mean([omega1, omega2])
                if omega_d_val < center_frequency:
                    transmission_below_center.append(N2_dB)
                else:
                    transmission_above_center.append(N2_dB)

            except:
                alpha2_solutions[i, j] = -25
                pass

        max_values_below_center.append(max(transmission_below_center) if transmission_below_center else -40)
        max_values_above_center.append(max(transmission_above_center) if transmission_above_center else -40)
        max_values.append(np.max(alpha2_solutions[:, j]))

        ax.plot(frequencies/1e9, [attenuation] * len(frequencies), alpha2_solutions[:, j], color=colors[j], lw=3.0)

    ax.set_xlim([min(frequencies)/1e9, max(frequencies)/1e9])  # Set the limit for the x-axis (omega_d)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.set_xlabel(r' Frequency (GHz)', fontsize=25)
    ax.set_ylabel('Symmetric Attenuation (dB)', fontsize=25)
    ax.set_zlabel(r'$S_{21}$ (dB)', fontsize=25)
    ax.view_init(15, 70)  # Adjust viewing angle for better visualization

    # Increase tick label size and add padding
    ax.tick_params(axis='x', which='major', labelsize=25, pad=15)  # Adjust for x-axis
    ax.tick_params(axis='y', which='major', labelsize=25, pad=15)  # Adjust for y-axis
    ax.tick_params(axis='z', which='major', labelsize=25, pad=15)  # Adjust for z-axis

    # Increase spacing between numbers and axis labels
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20

    # Remove grid lines (this approach disables the grid but keeps the pane, adjust as necessary)
    ax.grid(False)
    ax.xaxis._axinfo["grid"]['color'] = (0,0,0,0)
    ax.yaxis._axinfo["grid"]['color'] = (0,0,0,0)
    ax.zaxis._axinfo["grid"]['color'] = (0,0,0,0)

    # If you also want to remove the background pane for a cleaner look, you can do so:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Increase spacing between numbers and axis labels
    ax.xaxis.labelpad = 30 # Increase spacing for x-axis label
    ax.yaxis.labelpad = 30 # Increase spacing for y-axis label
    ax.zaxis.labelpad = 30 # Increase spacing for z-axis label

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/S21_waterfall_plot_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.png', dpi=300)
    plt.savefig(f'{folder_name}/S21_waterfall_plot_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.pdf', dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(attenuations, max_values_below_center, marker='o', linestyle='-', color='b', label='left peak')
    plt.plot(attenuations, max_values_above_center, marker='o', linestyle='-', color='crimson', label='right peak')
    plt.xlabel('Symmetric Attenuation (dB)', fontsize=20)
    plt.ylabel(r'$S_{21}^{\rm{max}}$(dB)', fontsize=20)
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/increasing_photon_numbers_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.png')
    plt.savefig(f'{folder_name}/increasing_photon_numbers_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.pdf')
    plt.close()

    # Plot for alpha2
    plt.figure(figsize=(7, 7))
    plt.imshow(alpha2_solutions.T, extent=[frequencies[0]/1e9, frequencies[-1]/1e9,
                                        attenuations[-1], attenuations[0]],
            aspect='auto', cmap='inferno')#, vmin=-35.1165)#, vmax=vmaxs[phase])

    clb = plt.colorbar()
    clb.set_label(label=r"$S_{21}$ [dB]", size=17)

    plt.xlabel("Frequency [GHz]", fontsize=20)
    plt.ylabel("Symmetric Attenuation [dB]", fontsize=20)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))

    clb.ax.tick_params(labelsize=20) 
    
    plt.gca().tick_params('both', labelsize=20)

    plt.tight_layout()
    plt.savefig(f'{folder_name}/nl_hopping_optimized_parameters_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.png', dpi=300)
    plt.savefig(f'{folder_name}/nl_hopping_optimized_parameters_phase_{phase:.2f}_epsilon_dbm_{epsilon_dBm:.2f}.pdf', dpi=300)
    plt.close()

alpha = [2000, 0, 2000, 0]
phases = [0, np.pi]
attenuations = np.linspace(0, 15, 31)

epsilon_dBms = [-10]
frequencies_to_simulate = {0: np.linspace(5.98e9, 6.09e9, 1000),
                           np.pi: np.linspace(6.015e9, 6.055e9, 1000)}

for epsilon_dBm in epsilon_dBms:
  for phase in phases:
    frequencies = frequencies_to_simulate[phase]
    create_plots_for_phase(frequencies, attenuations, phase, epsilon_dBm=epsilon_dBm)
    print(f'Phase: {phase} and epsilon_dBm: {epsilon_dBm} done')
