import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from matplotlib import ticker

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
J_0 = parameters["J0"] * 1e6

h_bar = 1.054571817e-34
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']
alpha_sat = P_sat / (h_bar * omega1 * kappa_c)

plot_individual = False

def frequency_LC(phase, J_0):
    kappa_int = np.mean([kappa_int_1, kappa_int_2])
    kappa_in_out = np.mean([kappa_drive, kappa_readout])

    num = 2 * (kappa_c + kappa_int + kappa_in_out) * np.cos(phase/2) 
    den = 1 + np.sin(phase/2)

    freq = num/den + J_0*np.cos(phase/2)

    return np.abs(freq)

# Load the data from the CSV
df = pd.read_csv('time_evolution_LC/results.csv')

# Extract unique phase and gain values
phase_values = df['Phase [rads.]'].unique()
gain_values = df['Gain [dB]'].unique()
phase_values.sort()
gain_values.sort()

# Create a grid for simulated amplitudes
final_freq_matrix = np.full((len(phase_values), len(gain_values)), np.nan)

# Populate the matrix with data from the dataframe, setting conditions for frequency
for index, row in df.iterrows():
    if row['Frequency [Hz]'] != 0.0:  # Only process non-zero frequencies
        phase_index = np.where(phase_values == row['Phase [rads.]'])[0][0]
        gain_index = np.where(gain_values == row['Gain [dB]'])[0][0]
        frequency_LC_simulated = row['Frequency [Hz]']
        ## Frequency [MHz]
        final_freq_matrix[phase_index, gain_index] = np.abs(frequency_LC_simulated)/1e6 if frequency_LC_simulated != 0 else np.nan

# Analytical amplitude calculations using the function
gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)
analytical_frequency = frequency_LC(phase_grid, J_0)/1e6

# Calculate differences in photon numbers
relative_difference_freq = np.abs(final_freq_matrix - analytical_frequency)

# Plotting Relative difference
plt.figure(figsize=(7.5, 6))
contour_plot = plt.contourf(gain_values, phase_values, relative_difference_freq, levels=800, cmap='inferno')
cbar = plt.colorbar(contour_plot)
# Format colorbar ticks to show fewer decimal places
cbar.formatter = ticker.FormatStrFormatter(r'$%.2f$')  # Adjust the number of decimal places here
cbar.set_label(r'$||\delta \omega_{LC}^{\rm{{num}}}| - |\delta \omega_{LC}^{\rm{{th}}}||$  [MHz]')
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # Set the number of bins to 5 or any preferred number
cbar.update_ticks()

# Set major locator to MultipleLocator (pi)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)


plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('relative_diff_freq.png', dpi=400)
plt.close()

## Example calculation

# # 8.4,1.9883497807530337,863861151023671.5,15210025.81022543
# simple_calculation = frequency_LC(1.9883497807530337, J_0)
# freq_simulated = 15210025.81022543/1e6

# print(simple_calculation)
# print(freq_simulated)

# print(np.abs(freq_simulated - simple_calculation), ' MHz')
# print(np.abs(freq_simulated - simple_calculation)*1e3, ' kHz')
