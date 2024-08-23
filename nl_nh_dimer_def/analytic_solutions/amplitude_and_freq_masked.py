import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from numpy import arange, floor, ceil
from matplotlib.ticker import MaxNLocator

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

def format_colorbar(cbar, tick_size=20, num_ticks=5, label_size=20):
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

def amplitude_LC(gain, phase):
    ## this doesn't care about J_0
    kappa_int = np.mean([kappa_int_1, kappa_int_2])
    kappa_in_out = np.mean([kappa_drive, kappa_readout])

    num = (10**(gain/20) * (1 + np.sin(phase/2)) * (kappa_c**2 * omega1 * h_bar * alpha_sat + kappa_c * b_amp)) - 2*b_amp*(kappa_c + kappa_in_out + kappa_int)
    den = 2*kappa_c*omega1*h_bar*(kappa_c + kappa_in_out + kappa_int)
    N2 = num/den
    return N2

def frequency_LC(phase, J_0):
    kappa_int = np.mean([kappa_int_1, kappa_int_2])
    kappa_in_out = np.mean([kappa_drive, kappa_readout])

    num = 2 * (kappa_c + kappa_int + kappa_in_out) * np.cos(phase/2) 
    den = 1 + np.sin(phase/2)

    freq = num/den + J_0*np.cos(phase/2)

    return np.abs(freq)
    # return freq

gain_values = np.linspace(4.0, 8.4, 1000)
phase_values = np.linspace(0, 2*np.pi, 1000)  # Phase from 0 to 2*pi

gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)
amplitude_grid = amplitude_LC(gain_grid, phase_grid)

### Analytic Amplitude of the Limit Cycle
## baseline experimental data
power_dBm = -42
power_watts = 10**(power_dBm / 10) * 1e-3

# Calculating photon numbers
photon_numbers_base = power_watts / (kappa_readout * h_bar * omega2)
masked_amplitude_grid = np.where(amplitude_grid > alpha_sat, amplitude_grid, photon_numbers_base)
# masked_amplitude_grid = np.where(amplitude_grid > alpha_sat, amplitude_grid, 0)

# Calculate power in dBm
photon_numbers = masked_amplitude_grid
power_watts = photon_numbers * kappa_readout * h_bar * omega2
power_dBm = 10 * np.log10(power_watts * 1e3)

threshold = 4.7822 ### dB. Measured from experimental data.

# plt.figure(figsize=(10, 8))
plt.figure(figsize=(7.5, 6))

# Plot power in dBm with contourf
cf = plt.contourf(gain_grid, phase_grid, power_dBm, levels=200, cmap='inferno')
cbar_amp = plt.colorbar(cf, label='Amplitude [dBm]')
format_colorbar(cbar_amp)

# Overlay a crimson contour to indicate regions where amplitude exceeds alpha_sat
contour_levels = [alpha_sat]
contours = plt.contour(gain_grid, phase_grid, amplitude_grid, levels=contour_levels, colors='crimson', linewidths=3.0)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().axvline(x=threshold, ls='--', lw=3.0, color='crimson')
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('analytic_amplitude_limit_cycle_masked.png', dpi=400)
plt.close()

##### Analytic Frequency of the Limit Cycle
# Calculate frequency grid
frequency_grid = frequency_LC(phase_grid, J_0)/1e6

# # Create a masked frequency grid where amplitude is greater than alpha_sat
masked_frequency_grid = np.where(amplitude_grid > alpha_sat, frequency_grid, np.nan)

# Plotting the masked frequency grid
plt.figure(figsize=(7.5, 6))
cf_freq = plt.contourf(gain_grid, phase_grid, masked_frequency_grid, levels=200, cmap='inferno')
cbar_freq = plt.colorbar(cf_freq, label=r'$|\Delta \omega_{LC}|$ [MHz]')
# cbar_freq = plt.colorbar(cf_freq, label=r'$\Delta \omega_{LC}$ [MHz]')
locator = MaxNLocator(nbins=4)  # Adjust 'nbins' to change number of ticks
cbar_freq.locator = locator
cbar_freq.update_ticks()
cbar_freq.ax.tick_params(labelsize=20)

# contour_levels = [alpha_sat]
contours = plt.contour(gain_grid, phase_grid, amplitude_grid, levels=contour_levels, colors='crimson', linewidths=3.0)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().axvline(x=threshold, ls='--', lw=3.0, color='crimson')
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('analytic_freq_limit_cycle_masked.png', dpi=400)
plt.close()
