import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from matplotlib import ticker

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

J_0 = parameters["J0"] * 1e6
h_bar = 1.054571817e-34
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']
alpha_sat = P_sat / (h_bar * omega1 * kappa_c)

plot_individual = False

def amplitude_LC(gain, phase):
    kappa_int = np.mean([kappa_int_1, kappa_int_2])
    kappa_in_out = np.mean([kappa_drive, kappa_readout])
    num = (10**(gain/20) * (1 + np.sin(phase/2)) * (kappa_c**2 * omega1 * h_bar * alpha_sat + kappa_c * b_amp)) - 2*b_amp*(kappa_c + kappa_in_out + kappa_int)
    den = 2*kappa_c*omega1*h_bar*(kappa_c + kappa_in_out + kappa_int)
    return num/den

# Load the data from the CSV
df = pd.read_csv('time_evolution_LC/results.csv')

# Extract unique phase and gain values
phase_values = df['Phase [rads.]'].unique()
gain_values = df['Gain [dB]'].unique()
phase_values.sort()
gain_values.sort()

# Create a grid for simulated amplitudes
final_norm_matrix = np.full((len(phase_values), len(gain_values)), np.nan)  # Initialize with NaN

# Populate the matrix with data from the dataframe, setting conditions for frequency
for index, row in df.iterrows():
    if row['Frequency [Hz]'] != 0.0:  # Only process non-zero frequencies
        phase_index = np.where(phase_values == row['Phase [rads.]'])[0][0]
        gain_index = np.where(gain_values == row['Gain [dB]'])[0][0]
        photon_numbers = row['Amplitude [dBm]']  # Assuming this column represents photon numbers in dBm directly
        final_norm_matrix[phase_index, gain_index] = photon_numbers

# Analytical amplitude calculations using the function
gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)
analytical_amplitude = amplitude_LC(gain_grid, phase_grid)

# Calculate differences in photon numbers
relative_difference_pn = (np.abs(final_norm_matrix - analytical_amplitude)/final_norm_matrix) * 100  # Transposed to match dimensions

# Find indices where relative error exceeds 50%
indices_exceeding_50 = np.argwhere(relative_difference_pn > 50)
print(indices_exceeding_50)

# Extract actual gain and phase values along with the relative error percentages
exceeding_data_points = [(phase_values[row], gain_values[col], relative_difference_pn[row, col]) for row, col in indices_exceeding_50]
print(exceeding_data_points)

### Removed a single point outlier in the numerical simulations
# Indices: [[17 47]]
# Values: phase, gain, relative_difference [(1.352077850912063, 6.6177215189873415, 143.28566783754383)]

relative_difference_pn[relative_difference_pn > 50] = np.nan

### Now, error analysis in power.
power_watts_analytical = analytical_amplitude * kappa_readout * h_bar * omega2
power_dBm_analytical = 10 * np.log10(power_watts_analytical * 1e3)

power_watts_simulated = final_norm_matrix * kappa_readout * h_bar * omega2
power_dBm_simulated = 10 * np.log10(power_watts_simulated * 1e3)

absolute_difference_dBm = np.abs(power_dBm_simulated - power_dBm_analytical)

# Find indices where relative error exceeds 50%
indices_exceeding_50 = np.argwhere(absolute_difference_dBm > 3)
print(indices_exceeding_50)

# Extract actual gain and phase values along with the relative error percentages
exceeding_data_points = [(phase_values[row], gain_values[col], relative_difference_pn[row, col]) for row, col in indices_exceeding_50]
print(exceeding_data_points)

absolute_difference_dBm[absolute_difference_dBm > 3] = np.nan

# Plotting Relative difference
plt.figure(figsize=(7.5, 6))
contour_plot = plt.contourf(gain_values, phase_values, relative_difference_pn, levels=800, cmap='inferno')
cbar = plt.colorbar(contour_plot)
cbar.set_label(r'$|n_{\rm{{LC}}}^{\rm{{sim}}} - n_{\rm{{LC}}}^{\rm{{th}}}| / n_{\rm{{LC}}}^{\rm{{sim}}} \ \ [\%]$ ')
cbar.formatter = ticker.FormatStrFormatter(r'$%.2f$')  # Adjust the number of decimal places here
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # Set the number of bins to 5 or any preferred number
# Update the ticks after setting the new locator
cbar.update_ticks()

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)

plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('relative_diff_amp.png', dpi=400)
plt.close()

# Plotting Absolute difference dBm
plt.figure(figsize=(7.5, 6))
contour_plot = plt.contourf(gain_values, phase_values, absolute_difference_dBm, levels=800, cmap='inferno')
cbar = plt.colorbar(contour_plot)

cbar.set_label(r'$|\rm{{Amp}}^{\rm{{sim}}} - \rm{{Amp}}^{\rm{{th}}}| \ $ [dB]')
cbar.formatter = ticker.FormatStrFormatter(r'$%.2f$')  # Adjust the number of decimal places here
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # Set the number of bins to 5 or any preferred number
cbar.update_ticks()

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)


plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('relative_diff_amp_dBm.png', dpi=400)
plt.close()

if plot_individual:

    for i, gain in enumerate(gain_values):
        plt.figure(figsize=(10, 4))
        plt.plot(phase_values, relative_difference_pn[:, i], label=f'Gain: {gain} dB')
        plt.xlabel('Phase [rads]')
        plt.ylabel('Relative Error [%]')
        plt.title(f'Relative Error in Photon Numbers at Gain {gain} dB')
        plt.legend()

        gain_dir = f'constant_gains/'
        os.makedirs(gain_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(gain_dir, f'relative_error_gain_{gain}.png'))
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(phase_values, absolute_difference_dBm[:, i], label=f'Gain: {gain} dB')
        plt.xlabel('Phase [rads]')
        plt.ylabel('Absolute Error [dB]')
        plt.title(f'Relative Error in dBm at Gain {gain} dB')
        plt.legend()

        # Directory for this gain
        gain_dir = f'constant_gains_dBm/'
        os.makedirs(gain_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(gain_dir, f'relative_error_gain_{gain}.png'))
        plt.close()
