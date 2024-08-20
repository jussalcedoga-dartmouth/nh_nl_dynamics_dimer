import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from matplotlib import cm
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.font_manager as fm
from scipy.optimize import root, approx_fprime

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

# plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

threshold = 4.52

with open('params.json', 'r') as f:
    parameters = json.load(f)

### assign the parameters accordingly...
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
grid_phase_diagram = 200

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

def f(phi):
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

### Gain of the amplifier in linear operation
params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c) # alpha_sat
assert h_bar * omega1 * alpha_sat * kappa_c == P_sat

## implementation for J_nl as it is written in the notes.
def J_nl(alpha, net_gain):
    prefactor = kappa_c * 10 ** (net_gain/20)
    if alpha <= alpha_sat:
        return prefactor * 1.0
    else:
        numerator = b_amp + h_bar * omega2 * alpha_sat * kappa_c
        denominator = b_amp + h_bar * omega2 * alpha * kappa_c
        return prefactor * (numerator/denominator)

def func(alpha, phase, net_gain):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    d_alpha1 = -(1j*(omega1 - omega1) + kappa_diag_1)*alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j* phase)*alpha2_c
    d_alpha2 = -(1j*(omega2 - omega2) + kappa_diag_2)*alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

eigenvalue_grid = grid_phase_diagram
net_gains = np.linspace(4.0, 8.4, eigenvalue_grid)
phase_range = np.linspace(0, 2 * np.pi, eigenvalue_grid)

# Create a grid for gain and phase
gain_grid, phase_grid = np.meshgrid(net_gains, phase_range)

#### Experimental figures...
bare_cavity = 6.027e9

# Preallocate arrays for storing eigenvalue components
eig1_real_gain_phase = np.zeros_like(gain_grid)
eig1_imag_gain_phase = np.zeros_like(gain_grid)

def jacobian_numerical(alpha, phase, gain, epsilon=1e-12):
    func_to_diff = lambda x: func(x, phase, gain)
    return approx_fprime(alpha, func_to_diff, epsilon)

def max_real_part_eigenvalues(eigenvalues):
    index_max_real = np.argmax(np.real(eigenvalues))
    return eigenvalues[index_max_real]

def max_imag_part_eigenvalues(eigenvalues):
    largest_real_eigenvalue = max_real_part_eigenvalues(eigenvalues)
    return np.imag(largest_real_eigenvalue)

# Function to find fixed points and compute the Jacobian at those points
def fixed_points(phase_val, gain, initial_guess):
    func_to_optimize = lambda x: func(x, phase_val, gain)
    sol = root(func_to_optimize, initial_guess, 
               jac=lambda x: jacobian_numerical(x, phase_val, gain), tol=1e-20, method='hybr')
    if sol.success:
        jacobian_at_sol = jacobian_numerical(sol.x, phase_val, gain)
        return sol.x, jacobian_at_sol
    else:
        return None, None

# Assume definitions for gain_grid, phase_range, and other variables are made elsewhere
for i in range(len(gain_grid)):
    for j in range(len(phase_range)):
        gain = gain_grid[i, j]
        phase = phase_grid[i, j]
        omega_d = bare_cavity # Fixed drive frequency for this plot

        initial_guess = [0, 0, 0, 0]  # Initial guess for the root-finding algorithm
        result, jac = fixed_points(phase, gain, initial_guess)

        if result is not None:
            eigs = np.linalg.eigvals(jac)
            eig1_imag_gain_phase[i, j] = (max_imag_part_eigenvalues(eigs)/1e6)
            eig1_real_gain_phase[i, j] = (max_real_part_eigenvalues(eigs).real/1e6)  # Also corrects this line to return only real part

### This is pretty important. In reality this corresponds to the point where gain balances loss.
net_gains = np.array(gain_grid)

# Function to format the colorbar
def format_colorbar(cbar, tick_size=30, num_ticks=5, label_size=30):
    cbar.ax.tick_params(labelsize=tick_size)
    # Set the number of ticks in the colorbar
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)

def set_yaxis_ticks(ax, tick_size=25):
    # Set major locator to MultipleLocator (pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
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

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # Increase tick size
    ax.tick_params(axis='y', which='major', labelsize=tick_size)

# # Create a mask where the real part of the eigenvalue is greater than 0
mask = eig1_real_gain_phase > 0

# Apply the mask to the imaginary part data
masked_ximag_data = np.ma.array(eig1_imag_gain_phase, mask=~mask)

plt.figure(figsize=(8, 6))
# Plotting the masked imaginary part
contour_eig3 = plt.contourf(net_gains, phase_grid, masked_ximag_data, levels=800, cmap='inferno')
contour_eig1 = plt.contour(net_gains, phase_grid, eig1_real_gain_phase, levels=[0], colors='crimson', linestyles='-', linewidths=2.0)

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',  # Load amsmath for math enhancements
})

threshold = 4.7822

cbar_eig3 = plt.colorbar(contour_eig3, label=r'$|\Delta \omega_{\pm}|$ [MHz]', ax=plt.gca())
format_colorbar(cbar_eig3)
set_yaxis_ticks(plt.gca(), tick_size=35)

plt.gca().axvline(x=threshold, ls='--', lw=2.0, color='crimson')

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
plt.gca().tick_params(axis='x', which='major', labelsize=25)

plt.gca().set_xlabel(r'$\Delta G$ [dB]', fontsize=25)
plt.gca().set_ylabel(r'$\phi$', fontsize=35)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}",  # To enable bold math with \bm{}
})

helvetica_font = fm.FontProperties(family='Helvetica', weight='bold')

# Save the figure
plt.savefig(f'freq_lc_from_eigenvalues_J0_{J0/1e6:.2f}MHz.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.close()
