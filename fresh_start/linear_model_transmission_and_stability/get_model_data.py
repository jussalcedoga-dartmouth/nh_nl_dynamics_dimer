import numpy as np
from scipy.optimize import root, fsolve, root_scalar, least_squares
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import pandas as pd

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

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
epsilon_dBm = -30

# Frequency ranges for different models
nh_num_points = 200
hermitian_num_points = 200
num_net_gains = 31

transmission_notes = True

def J(net_gain):
    return 10**((net_gain) / 20) * kappa_c

def kappa_T(net_gain, kappa_0):
    return 2*kappa_0 - J(net_gain)

def f(phi):
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

## Dynamical Matrix - We can get steady state solutions from here.
def matrix_A(net_gain, phi, omega_d):
    kT1 = kappa_T(net_gain, kappa_T_1)
    kT2 = kappa_T(net_gain, kappa_T_2)

    J_val = J(net_gain)

    return np.array([[-(1j * (omega1 - omega_d) + kT1), (-1j * J_val - f(phi)) * np.exp(-1j * phi)],
                     [(-1j*J_val - f(phi)), -(1j * (omega2 - omega_d) + kT2)]])

def matrix_A_no_drive(net_gain, phi):
    kT1 = kappa_T(net_gain, kappa_T_1)
    kT2 = kappa_T(net_gain, kappa_T_2)
    J_val = J(net_gain)

    return np.array([[-(1j * (omega1) + kT1), (-1j * J_val - f(phi)) * np.exp(-1j * phi)],
                     [(-1j * J_val - f(phi)), -(1j * (omega2) + kT2)]])

## Transmission solution directly from the dynamical matrix
def get_data_for_phase(frequencies, net_gains, phase, epsilon_dBm):
    alpha2_solutions = np.empty((len(net_gains), len(frequencies)))

    for j, net_gain in enumerate(net_gains):
        for i, omega_d_val in enumerate(frequencies):

            A_inv = np.linalg.inv(matrix_A(net_gain, phase, omega_d_val))

            epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
            epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d_val))

            alpha2_sol = np.dot(A_inv, np.array([epsilon, 0]))

            ## Then only select the solutions for photon numbers in cavity 2.
            N2_total = np.abs(alpha2_sol[1])**2
            
            ## This is the transmission expression reported in the document.
            N2_dB = 10*np.log10(((omega2 * kappa_drive * kappa_readout) / omega_d_val) * (N2_total / epsilon**2))
            alpha2_solutions[j, i] = N2_dB

    return alpha2_solutions, net_gains

def get_metrics_spectrum_theory(alpha2_solutions, frequencies, net_gains, center_freq=6.028e9):
    results = []

    for i, net_gain in enumerate(net_gains):
        transmissions = alpha2_solutions[i, :]
        mask_below = frequencies < center_freq
        mask_above = frequencies > center_freq

        if np.any(mask_below) and np.any(mask_above):
            max_index_below = np.argmax(transmissions[mask_below])
            max_index_above = np.argmax(transmissions[mask_above]) + np.sum(mask_below)

            freq_max_below = frequencies[mask_below][max_index_below]
            freq_max_above = frequencies[mask_above][max_index_above - np.sum(mask_below)]

            difference = np.abs(freq_max_below - freq_max_above) * 1e3
            max_value_below = transmissions[mask_below][max_index_below]
            max_value_above = transmissions[mask_above][max_index_above - np.sum(mask_below)]

            results.append((net_gain, difference, max_value_below, max_value_above))

    results.sort(key=lambda x: x[0])
    if results:
        net_gains, differences, max_values_below, max_values_above = zip(*results)
        return net_gains, differences, max_values_below, max_values_above
    else:
        return [], [], [], []
    
def get_metrics_spectrum_nonhermitian_theory(alpha2_solutions, frequencies, net_gains):
    results = []

    for i, net_gain in enumerate(net_gains):
        transmissions = alpha2_solutions[i, :]
        max_index = np.argmax(transmissions)
        max_transmission = transmissions[max_index]

        results.append((net_gain, max_transmission))

    results.sort(key=lambda x: x[0])
    if results:
        net_gains, max_transmissions = zip(*results)
        return net_gains, max_transmissions
    else:
        return [], [], [], []

def get_phase_diagram(ax):
    csv_filename = 'phase_diagram_data.csv'
    loaded_df = pd.read_csv(csv_filename)

    # Reshape the data for plotting
    gain_values = loaded_df['Gain [dB]'].unique()
    phase_values = loaded_df['Phase [rads.]'].unique()
    amplitude_matrix_reshaped = loaded_df.pivot('Phase [rads.]', 'Gain [dB]', 'Amplitude [dBm]').values
    frequency_matrix_reshaped = loaded_df.pivot('Phase [rads.]', 'Gain [dB]', 'Frequency [GHz]').values

    # Define the condition to find the outlier
    outlier_condition = (loaded_df['Amplitude [dBm]'] > -40) & (loaded_df['Gain [dB]'] < 6.3) & (loaded_df['Phase [rads.]'] > 5.0)

    # Identify the phase index of the outlier
    outlier_phase = loaded_df[outlier_condition]['Phase [rads.]'].unique()
    if len(outlier_phase) > 0:
        outlier_phase = outlier_phase[0]  # Assume only one outlier phase to simplify

    # Remove the outlier by setting to NaN and interpolate to fill in
    for index in range(len(phase_values)):
        if phase_values[index] == outlier_phase:
            amplitude_matrix_reshaped[index, :] = np.nan
            frequency_matrix_reshaped[index, :] = np.nan

    # Interpolate NaN values
    amplitude_matrix_reshaped = pd.DataFrame(amplitude_matrix_reshaped).interpolate().values
    frequency_matrix_reshaped = pd.DataFrame(frequency_matrix_reshaped).interpolate().values

    unique_gains = np.linspace(4, 8.4, 200)
    unique_phases = np.linspace(0, 2*np.pi, 200)
    stability_matrix = np.zeros((len(unique_gains), len(unique_phases)))

    for i, net_gain in enumerate(unique_gains):
        for j, phi in enumerate(unique_phases):
            A = matrix_A_no_drive(net_gain, phi)

            ## compute the eigenvalues of the matrix A
            eigenvalues = np.linalg.eigvals(A)
            stability_matrix[i, j] = np.max(eigenvalues.real)

    contourf_plot = ax.contourf(gain_values, phase_values, amplitude_matrix_reshaped, levels=200, cmap='inferno')
    contour_lines = ax.contour(unique_gains, unique_phases, stability_matrix.T, levels=[0], colors='white', linewidths=3.0)
    cbar = plt.colorbar(contourf_plot, ax=ax)
    cbar.set_ticks(np.linspace(contourf_plot.get_clim()[0], contourf_plot.get_clim()[1], num=5))
    cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])
    cbar.set_label('Amplitude [dBm]')
    ax.axvline(x=4.7822, ls='--', lw=3.0, color='crimson')
    ax.set_title('Stability Phase Diagram')
    ax.set_xlabel('Net Gain [dB]')
    ax.set_ylabel(r'$\phi$ [rad]')

def get_used_parameters():
    return omega1/1e9, omega2/1e9, kappa_drive/1e6, kappa_readout/1e6, kappa_int_1/1e6, kappa_int_2/1e6, kappa_c/1e6, beta/1e6, reflections_amp, J0/1e6