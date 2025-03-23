import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from numpy import arange, floor, ceil

def clippedcolorbar(CS, label='', **kwargs):

    fig = plt.gcf()  # Get the current figure
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim((vmin, vmax))
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin < vmin
    clipupper = CS.zmax > vmax
    noextend = 'extend' in kwargs and kwargs['extend'] == 'neither'
    
    # Set the colorbar boundaries
    boundaries = arange((floor(vmin / step) - 1 + 1 * (cliplower and noextend)) * step, 
                        (ceil(vmax / step) + 1 - 1 * (clipupper and noextend)) * step, step)
    
    kwargs['boundaries'] = boundaries

    # If the z-values are outside the colorbar range, add extend marker(s)
    if not ('extend' in kwargs) or kwargs['extend'] in ['min', 'max']:
        extend_min = cliplower or ('extend' in kwargs and kwargs['extend'] == 'min')
        extend_max = clipupper or ('extend' in kwargs and kwargs['extend'] == 'max')
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    
    cbar = fig.colorbar(m, **kwargs)
    cbar.set_label(label)  # Set the label for the colorbar
    return cbar

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
threshold = 4.7822 ### dB. Measured from experimental data.

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
    photon_numbers = row['Amplitude [dBm]']
    power_watts = photon_numbers * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)
    final_norm_matrix[phase_index, gain_index] = power_dBm

stability_matrix = np.zeros((len(gain_values), len(phase_values)))

for i, net_gain in enumerate(gain_values):
    for j, phi in enumerate(phase_values):
        A = matrix_A_no_drive(net_gain, phi)
        eigenvalues = np.linalg.eigvals(A)
        stability_matrix[i, j] = np.max(eigenvalues.real)

# Plotting using imshow
# plt.figure(figsize=(8, 7))
plt.figure(figsize=(7.5, 6))

vmin = -43

contourf_plot = plt.contourf(gain_values, phase_values, final_norm_matrix, levels=800, cmap='inferno', vmin=vmin)
contour_lines = plt.contour(gain_values, phase_values, stability_matrix.T, levels=[0], colors='crimson', linewidths=3.0)

# # Create colorbar and specify label
cbar = clippedcolorbar(contourf_plot, label=r'Amplitude [dBm]')

# Set major locator to MultipleLocator (pi)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().tick_params(axis='y', which='major', labelsize=30)
plt.gca().axvline(x=threshold, ls='--', lw=3.0, color='crimson')
plt.xlabel(r'$\Delta G$ [dB]', fontsize=30)
plt.ylabel(r'$\phi$', fontsize=35)
plt.tight_layout()
plt.savefig('numerical_amplitude_limit_cycle.png', dpi=400)
plt.close()
