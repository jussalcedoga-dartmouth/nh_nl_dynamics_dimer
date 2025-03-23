import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from numpy import arange, floor, ceil
from matplotlib.ticker import MaxNLocator

def format_colorbar(cbar, tick_size=30, num_ticks=5, label_size=30):
    cbar.ax.tick_params(labelsize=tick_size)
    # Set the number of ticks in the colorbar
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

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

## Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

def J(net_gain):
    return 10**((net_gain) / 20) * kappa_c

def kappa_T(net_gain, kappa_0):
    return 2*kappa_0 - J(net_gain)

def f(phi):
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

def matrix_A_no_drive(net_gain, phi):
    kT1 = kappa_T(net_gain, kappa_T_1)
    kT2 = kappa_T(net_gain, kappa_T_2)
    J_val = J(net_gain)

    return np.array([[-(1j * (omega1) + kT1), (-1j * J_val - f(phi)) * np.exp(-1j * phi)],
                     [(-1j * J_val - f(phi)), -(1j * (omega2) + kT2)]])

# Load results
df = pd.read_csv('time_evolution_LC/results.csv')

# Prepare grid for plotting
phase_values = df['Phase [rads.]'].unique()
gain_values = df['Gain [dB]'].unique()
phase_values.sort()
gain_values.sort()

# Prepare data for contour plot
final_norm_matrix = np.zeros((len(phase_values), len(gain_values)))
for index, row in df.iterrows():
    phase_index = np.where(phase_values == row['Phase [rads.]'])[0][0]
    gain_index = np.where(gain_values == row['Gain [dB]'])[0][0]

    # convert photon numbers to power in dBm
    frequency_lc = row['Frequency [Hz]']
    # final_norm_matrix[phase_index, gain_index] = np.abs(frequency_lc)/1e6
    
    # final_norm_matrix[phase_index, gain_index] = np.abs(frequency_lc)/1e6 if frequency_lc != 0 else np.nan
    final_norm_matrix[phase_index, gain_index] = frequency_lc/1e6 if frequency_lc != 0 else np.nan

stability_matrix = np.zeros((len(gain_values), len(phase_values)))

for i, net_gain in enumerate(gain_values):
    for j, phi in enumerate(phase_values):
        A = matrix_A_no_drive(net_gain, phi)
        eigenvalues = np.linalg.eigvals(A)
        stability_matrix[i, j] = np.max(eigenvalues.real)

# Plotting using imshow
plt.figure(figsize=(7.5, 6))
# from matplotlib import pyplot as plt
contourf_plot = plt.contourf(gain_values, phase_values, final_norm_matrix, levels=1000, cmap='inferno')
contour_lines = plt.contour(gain_values, phase_values, stability_matrix.T, levels=[0], colors='crimson', linewidths=3.0)

threshold = 4.7822 ### dB. Measured from experimental data.

cbar_freq = plt.colorbar(contourf_plot, label=r'$\Delta \omega_{LC}$  [MHz]', ax=plt.gca())
# format_colorbar(cbar_freq)
locator = MaxNLocator(nbins=6)  # Adjust 'nbins' to change number of ticks
cbar_freq.locator = locator
cbar_freq.update_ticks()
cbar_freq.ax.tick_params(labelsize=20)


plt.axvline(x=threshold, ls='--', lw=3.0, color='crimson')

# Set major locator to MultipleLocator (pi)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)

plt.xlabel(r'$\Delta G$ [dB]', fontsize=30)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('numerical_frequency_limit_cycle.png', dpi=400)
plt.close()
