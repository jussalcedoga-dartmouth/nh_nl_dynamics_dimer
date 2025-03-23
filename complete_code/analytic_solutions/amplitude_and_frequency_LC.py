import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from numpy import arange, floor, ceil

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

with open('params.json', 'r') as f:
    parameters = json.load(f)

omega1 = parameters["omega1"] * 1e9
omega2 = parameters["omega2"] * 1e9
kappa_drive = parameters["kappa_drive"] * 1e6
kappa_readout = parameters["kappa_readout"] * 1e6
kappa_int_1 = parameters["kappa_int_1"] * 1e6
kappa_int_2 = parameters["kappa_int_2"] * 1e6
kappa_c = parameters["kappa_c"] * 1e6
h_bar = 1.054571817e-34

params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']
alpha_sat = P_sat / (h_bar * omega1 * kappa_c)

def amplitude_LC(gain, phase):
    num = (10**(gain/20) * (1 + np.sin(phase/2)) * (kappa_c**2 * omega1 * h_bar * alpha_sat + kappa_c * b_amp)) - 2*b_amp*(kappa_c + kappa_readout + kappa_int_1)
    den = 2*kappa_c*omega1*h_bar*(kappa_c + kappa_drive + kappa_int_1)
    N2 = num/den
    return N2

def frequency_LC(gain, phase):
    num = 2 * (kappa_c + kappa_int_1 + kappa_readout) * np.cos(phase/2)
    den = 1 + np.sin(phase/2)
    freq = num/den
    return np.abs(freq)

gain_values = np.linspace(4.0, 8.4, 200)
phase_values = np.linspace(0, 2*np.pi, 200)  # Phase from 0 to 2*pi

gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)
amplitude_grid = amplitude_LC(gain_grid, phase_grid)

# Plotting amplitude with contour
# plt.figure(figsize=(10, 8))
plt.figure(figsize=(7.5, 6))
im = plt.imshow(amplitude_grid, extent=(gain_values.min(), gain_values.max(), phase_values.min(), phase_values.max()),
           aspect='auto', origin='lower', cmap='inferno')

threshold = 4.7822

contour_levels = [alpha_sat]
contours = plt.contour(gain_grid, phase_grid, amplitude_grid, levels=contour_levels, colors='white')
plt.gca().axvline(x=threshold, ls='--', lw=3.0, color='white')

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.colorbar(im, label=r'$|\alpha_2|^2$')
plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('analytic_amplitude_limit_cycle.png', dpi=400)
plt.close()

# Calculate frequency grid
frequency_grid = frequency_LC(gain_grid, phase_grid)/1e6
# Plotting frequency with the same contour
plt.figure(figsize=(7.5, 6))

im = plt.imshow(frequency_grid, extent=(gain_values.min(), gain_values.max(), phase_values.min(), phase_values.max()),
           aspect='auto', origin='lower', cmap='inferno')
contours = plt.contour(gain_grid, phase_grid, amplitude_grid, levels=contour_levels, colors='white')

cbar_freq = plt.colorbar(im, label=r'$\Delta \omega_{LC}$ [MHz]')
format_colorbar(cbar_freq)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.gca().axvline(x=threshold, ls='--', lw=3.0, color='white')
plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('analytic_freq_limit_cycle.png', dpi=400)
plt.close()
