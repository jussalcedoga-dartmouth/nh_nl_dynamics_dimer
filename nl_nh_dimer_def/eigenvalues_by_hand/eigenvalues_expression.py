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

def format_colorbar(cbar, tick_size=30, num_ticks=5, label_size=30):
    cbar.ax.tick_params(labelsize=tick_size)
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
threshold = 4.7822
plot_unmasked = False

def imag_component_eigs(gain, phase):
    kappa_c = 8.7e6
    J = 10**(gain/20) * kappa_c
    return np.abs(J * np.cos(phase/2) + J_0*np.cos(phase/2))

def phase_boundary(gain, phase):
    kappa_T = 2*(kappa_int_1 + kappa_drive + kappa_c)
    return ((kappa_T) / (1 + np.sin(phase/2))) * 10**(-gain/20)

kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

def J(net_gain):
    return 10**((net_gain) / 20) * kappa_c

def kappa_T(net_gain, kappa_0):
    return 2*kappa_0 - J(net_gain)

## matrix A implementing Juan's model.
def matrix_A(net_gain, phi):
    kT1 = kappa_T(net_gain, kappa_T_1)
    kT2 = kappa_T(net_gain, kappa_T_2)
    J_val = J(net_gain)

    return np.array([[-(1j * omega1 + kT1), -1j * J_val * np.exp(-1j * phi)],
                     [-1j * J_val, -(1j * omega2 + kT2)]])

gain_values = np.linspace(4.0, 8.4, 200)
phase_values = np.linspace(0, 2*np.pi, 200)  # Phase from 0 to 2*pi

gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)

stability_matrix = np.zeros((len(gain_values), len(phase_values)))

for i, net_gain in enumerate(gain_values):
    for j, phi in enumerate(phase_values):
        A = matrix_A(net_gain, phi)
        eigenvalues = np.linalg.eigvals(A)
        stability_matrix[i, j] = np.max(eigenvalues.real)

# Calculate frequency grid
frequency_grid = imag_component_eigs(gain_grid, phase_grid)/1e6
stability_grid = phase_boundary(gain_grid, phase_grid)

# Plotting frequency with the same contour
plt.figure(figsize=(7.5, 6))

im = plt.imshow(frequency_grid, extent=(gain_values.min(), gain_values.max(), phase_values.min(), phase_values.max()),
           aspect='auto', origin='lower', cmap='inferno')

cbar_freq = plt.colorbar(im, label=r'$|\Delta \omega_{LC}|$ [MHz]')
format_colorbar(cbar_freq)

contour_lines = plt.contour(gain_values, phase_values, stability_matrix.T, levels=[0], colors='white', linewidths=3.0)

if plot_unmasked:
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.gca().tick_params(axis='y', which='major', labelsize=30)
    plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
    plt.ylabel(r'$\phi$', fontsize=35)
    plt.tight_layout()
    plt.savefig('imag_component_eigs_with_boundary.png', dpi=400)
    plt.close()

### masking the frequency so that it looks as what we compute.
# Grid definitions
gain_values = np.linspace(4.0, 8.4, 200)
phase_values = np.linspace(0, 2*np.pi, 200)

gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)

# Calculate frequency and stability grids
frequency_grid = imag_component_eigs(gain_grid, phase_grid)/1e6
stability_grid = phase_boundary(gain_grid, phase_grid)

# Mask frequencies where the stability condition is not met
frequency_masked = np.ma.masked_where(stability_matrix.T < 0, frequency_grid)

# Plotting the masked frequency grid
plt.figure(figsize=(7.5, 6))

cf_freq = plt.contourf(gain_grid, phase_grid, frequency_masked, levels=200, cmap='inferno')
cbar_freq = plt.colorbar(cf_freq, label=r'$|\Delta \omega_{+}|$ [MHz]')
# Create a MaxNLocator object with desired number of ticks
locator = MaxNLocator(nbins=4)  # Adjust 'nbins' to change number of ticks
cbar_freq.locator = locator
cbar_freq.update_ticks()
cbar_freq.ax.tick_params(labelsize=20)

# Plot contour at stability boundary
contour_levels = [kappa_c]  # Adjust contour level to your phase boundary condition
contour_lines = plt.contour(gain_values, phase_values, stability_matrix.T, levels=[0], colors='crimson', linewidths=3.0)

# Axis settings
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.gca().axvline(x=threshold, ls='--', lw=2.0, color='crimson')
plt.xlabel(r'$\Delta G$ [dB]', fontsize=35)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('imag_component_eigs_with_boundary_masked.png', dpi=400)
plt.close()
