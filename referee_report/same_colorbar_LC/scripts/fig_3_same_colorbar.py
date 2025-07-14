import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import numpy as np
import matplotlib
import os
import pandas as pd
from matplotlib import cm
import numpy as np
import json
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.font_manager as fm
from scipy.optimize import root, approx_fprime
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from return_analytics import *
from numpy import arange, floor, ceil

matplotlib.use('Agg')

# plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

## create folder to save the plots
folder_name = '../../plots'
os.makedirs(folder_name, exist_ok=True)

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


def clippedcolorbar(CS, label='', **kwargs):
    fig = plt.gcf()  # Get the current figure
    # vmin, vmax = CS.get_clim()

    # print(vmin, vmax)

    vmin = -43 # dBm
    vmax = -2.180 #dBm 

    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim((vmin, vmax))
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin < vmin
    clipupper = CS.zmax > vmax
    noextend = 'extend' in kwargs and kwargs['extend'] == 'neither'
    
    # Set the colorbar boundaries
    boundaries = np.arange((np.floor(vmin / step) - 1 + 1 * (cliplower and noextend)) * step,
                           (np.ceil(vmax / step) + 1 - 1 * (clipupper and noextend)) * step, step)
    
    kwargs['boundaries'] = boundaries

    cbar = fig.colorbar(m, **kwargs)
    cbar.set_label(label)  # Set the label for the colorbar
    return cbar

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

threshold = 4.7822 ### dB. Measured from experimental data.

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34

def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

def f(phi):
    # return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)
    return 1j*J0*(np.cos(phi/2))*np.exp(1j*phi/2)

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
    
# Function to compute the Jacobian numerically
def jacobian_numerical(alpha, phase, gain, epsilon=1e-12):
    func_to_diff = lambda x: func(x, phase, gain)
    return approx_fprime(alpha, func_to_diff, epsilon)

def max_real_part_eigenvalues(eigenvalues):
    # eigenvalues = np.linalg.eigvals(jacobian)
    return np.max(np.real(eigenvalues))

def max_imag_part_eigenvalues(eigenvalues):
    # eigenvalues = np.linalg.eigvals(jacobian)
    return np.max(np.imag(eigenvalues))

#### Analytics...
def amplitude_LC(gain, phase):

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

    # return np.abs(freq)
    return freq

# Load data from CSV
data = pd.read_csv('../data/theory/results.csv')
net_gains = data['Gain [dB]'].unique()
phase_range = data['Phase [rads.]'].unique()

# Create a grid for gain and phase
gain_grid, phase_grid = np.meshgrid(net_gains, phase_range)

# Initialize the arrays with dimensions [len(gain_grid), len(phase_range)]
eig1_real_gain_phase = np.zeros_like(gain_grid)
eig1_imag_gain_phase = np.zeros_like(gain_grid)

## Imaginary Component of the eigenvalues
for i in range(len(gain_grid)):
    for j in range(len(phase_grid)):
        gain = gain_grid[i, j]
        phase = phase_grid[i, j]

        initial_guess = [0, 0, 0, 0]  # Initial guess for the root-finding algorithm
        result, jac = fixed_points(phase, gain, initial_guess)

        if result is not None:
            eigs = np.linalg.eigvals(jac)
            eig1_imag_gain_phase[i, j] = (max_imag_part_eigenvalues(eigs)/1e6)
            eig1_real_gain_phase[i, j] = (max_real_part_eigenvalues(eigs)/1e6)

##### Limit cycle amplitude
# Load results
data = pd.read_csv('../data/theory/results.csv')
df = data

# Prepare grid for plotting
phase_values_num = df['Phase [rads.]'].unique()
gain_values_num = df['Gain [dB]'].unique()
phase_values_num.sort()
gain_values_num.sort()

# Prepare data for contour plot
final_norm_matrix = np.zeros((len(phase_values_num), len(gain_values_num)))
final_freq_matrix = np.zeros((len(phase_values_num), len(gain_values_num)))

# # Create a mask where the real part of the eigenvalue is greater than 0
mask = eig1_real_gain_phase > 0
# Apply the mask to the imaginary part data
# masked_ximag_data = np.ma.array(eig1_imag_gain_phase, mask=~mask)
masked_ximag_data = np.ma.array(final_freq_matrix, mask=~mask)
##### Numerical imaginary

for index, row in df.iterrows():
    phase_index = np.where(phase_values_num == row['Phase [rads.]'])[0][0]
    gain_index = np.where(gain_values_num == row['Gain [dB]'])[0][0]

    # convert photon numbers to power in dBm
    photon_numbers = row['Amplitude [dBm]']
    power_watts = photon_numbers * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)
    final_norm_matrix[phase_index, gain_index] = power_dBm

    time_domain_LC_freq = row['Frequency [Hz]']

    # final_freq_matrix[phase_index, gain_index] = np.abs(time_domain_LC_freq)/1e6
    ## just removing the data that doesn't have oscillations from the original FFT.
    # final_freq_matrix[phase_index, gain_index] = np.abs(time_domain_LC_freq)/1e6 if time_domain_LC_freq != 0 else np.nan
    final_freq_matrix[phase_index, gain_index] = time_domain_LC_freq/1e6 if time_domain_LC_freq != 0 else np.nan

############ Experimental data
# Load the CSV file
csv_filename = '../data/experiment/phase_diagram_data.csv'
loaded_df = pd.read_csv(csv_filename)

# Reshape the data for plotting using pivot with keyword arguments
gain_values = loaded_df['Gain [dB]'].unique()
phase_values = loaded_df['Phase [rads.]'].unique()
amplitude_matrix_reshaped = loaded_df.pivot(index='Phase [rads.]', columns='Gain [dB]', values='Amplitude [dBm]').values
frequency_matrix_reshaped = loaded_df.pivot(index='Phase [rads.]', columns='Gain [dB]', values='Frequency [GHz]').values


vmin_amp,  vmax_amp  = -43.0, 1.9765915013938988
vmin_freq, vmax_freq = -32.52389458546332, 32.52389458546333


# Define the condition to find the outlier
outlier_condition = (loaded_df['Amplitude [dBm]'] > -40) & (loaded_df['Gain [dB]'] < 6.3) & (loaded_df['Phase [rads.]'] > 5.0)
outlier_phases = loaded_df[outlier_condition]['Phase [rads.]'].unique()

# Remove the outlier by setting to NaN and interpolate to fill in
for phase in outlier_phases:
    phase_index = np.where(phase_values == phase)[0]
    amplitude_matrix_reshaped[phase_index, :] = np.nan
    frequency_matrix_reshaped[phase_index, :] = np.nan

# Interpolate NaN values
amplitude_matrix_reshaped = pd.DataFrame(amplitude_matrix_reshaped, index=phase_values, columns=gain_values).interpolate().values
frequency_matrix_reshaped = pd.DataFrame(frequency_matrix_reshaped, index=phase_values, columns=gain_values).interpolate().values

# Create masks based on amplitude condition
amplitude_condition_mask = amplitude_matrix_reshaped > -40
frequency_matrix_masked = np.where(amplitude_condition_mask, frequency_matrix_reshaped, np.nan)

# Determine colorbar limits based only on the masked frequency array
min_freq = np.nanmin(frequency_matrix_masked)
max_freq = np.nanmax(frequency_matrix_masked)

print('Experimental Amplitude lims', np.nanmin(amplitude_matrix_reshaped), np.nanmax(amplitude_matrix_reshaped))
print('Experimental Frequency lims', (6.027 - np.nanmax(frequency_matrix_masked))*1e3,   (6.027 - np.nanmin(frequency_matrix_masked))*1e3)

delta_frequency_matrix = (6.027 - frequency_matrix_masked) * 1e3

# Function to format the colorbar
def format_colorbar(cbar, tick_size=40, num_ticks=5, label_size=40):
    cbar.ax.tick_params(labelsize=tick_size)
    # Set the number of ticks in the colorbar
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)

def set_yaxis_ticks(ax, tick_size=35):
    # Set major locator to MultipleLocator (pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
    # Define custom formatter to handle specific label formatting
    def format_func(value, pos):
        if np.isclose(value, 0, atol=0.1):
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


plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',  # Load amsmath for math enhancements
})

def compute_and_plot_zero_contour(ax, gains, phases, fixed_points, max_real_part_eigenvalues, color='white'):
    # Create a mesh grid for gain and phase values
    gain_grid, phase_grid = np.meshgrid(gains, phases)
    
    # Initialize arrays to store the real parts of eigenvalues
    real_parts = np.zeros_like(gain_grid)
    
    # Compute eigenvalues across the grid
    for i in range(gain_grid.shape[0]):
        for j in range(gain_grid.shape[1]):
            gain = gain_grid[i, j]
            phase = phase_grid[i, j]
            initial_guess = [0, 0, 0, 0]  # Default initial guess for the root-finding algorithm
            result, jac = fixed_points(phase, gain, initial_guess)
            if result is not None:
                eigs = np.linalg.eigvals(jac)
                real_parts[i, j] = max_real_part_eigenvalues(eigs)
    
    # Plot zero contour
    CS = ax.contour(gain_grid, phase_grid, real_parts, levels=[0], colors=color, linewidths=3.0)
    return CS

# ------------- PLOTTING ---------------------------------------------------
fig = plt.figure(figsize=(17, 20))

# 2 real columns (Amplitude | Frequency) ; titles live in their own row
gs = GridSpec(
    4, 2, figure=fig,
    height_ratios=[0.001, 1, 1, 1],   # << was 0.02
    wspace=0.40, hspace=0.28
)

# ── title row (axes just for the text) ------------------------------------
ax_t_amp  = fig.add_subplot(gs[0, 0]); ax_t_amp.axis("off")
ax_t_freq = fig.add_subplot(gs[0, 1]); ax_t_freq.axis("off")

# 2) drop the text a bit lower inside that row
ax_t_amp .text(0.50, -0.10, 'Amplitude', ha='center', va='top',  fontsize=40,
               transform=ax_t_amp.transAxes)
ax_t_freq.text(0.50, -0.10, 'Frequency', ha='center', va='top', fontsize=40,
               transform=ax_t_freq.transAxes)

# ── main axes -------------------------------------------------------------
# column 0 – amplitude
ax_amp_exp = fig.add_subplot(gs[1, 0])
ax_amp_num = fig.add_subplot(gs[2, 0])
ax_amp_ana = fig.add_subplot(gs[3, 0])

# column 1 – frequency
ax_freq_exp = fig.add_subplot(gs[1, 1])
ax_freq_num = fig.add_subplot(gs[2, 1])
ax_freq_ana = fig.add_subplot(gs[3, 1])

# ── Amplitude column ------------------------------------------------------
c_amp_exp = ax_amp_exp.contourf(
    gain_values, phase_values, amplitude_matrix_reshaped,
    levels=500, cmap='inferno', vmin=vmin_amp, vmax=vmax_amp
)
compute_and_plot_zero_contour(ax_amp_exp, gain_values, phase_values,
                              fixed_points, max_real_part_eigenvalues)
set_yaxis_ticks(ax_amp_exp)

c_amp_num = ax_amp_num.contourf(
    gain_values_num, phase_values_num, final_norm_matrix,
    levels=800, cmap='inferno', vmin=vmin_amp, vmax=vmax_amp
)
compute_and_plot_zero_contour(ax_amp_num, gain_values_num, phase_values_num,
                              fixed_points, max_real_part_eigenvalues)
set_yaxis_ticks(ax_amp_num)

# analytic amplitude – remove its internal bar
before = len(fig.axes)
create_amplitude_plot(ax_amp_ana, vmin_amp, vmax_amp)
for extra in fig.axes[before:]:
    if extra not in [ax_amp_ana]:
        extra.remove()

# ── Frequency column -------------------------------------------------------
c_freq_exp = ax_freq_exp.contourf(
    gain_values, phase_values, delta_frequency_matrix,
    levels=500, cmap='inferno', vmin=vmin_freq, vmax=vmax_freq
)
compute_and_plot_zero_contour(ax_freq_exp, gain_values, phase_values,
                              fixed_points, max_real_part_eigenvalues, 'crimson')

c_freq_num = ax_freq_num.contourf(
    net_gains, phase_range, masked_ximag_data,
    levels=500, cmap='inferno', vmin=vmin_freq, vmax=vmax_freq
)
compute_and_plot_zero_contour(ax_freq_num, net_gains, phase_range,
                              fixed_points, max_real_part_eigenvalues, 'crimson')

before = len(fig.axes)
create_frequency_plot(ax_freq_ana, vmin_freq, vmax_freq)
for extra in fig.axes[before:]:
    if extra not in [ax_freq_ana]:
        extra.remove()

# ── Shared colour-bars -----------------------------------------------------
# compute exact positions from the panel bounding-boxes
def add_vertical_colorbar(mappable, ax_top, ax_bottom, label, left_shift):
    pos_top    = ax_top.get_position()
    pos_bottom = ax_bottom.get_position()
    bar_left   = pos_top.x1 + left_shift          # just to the right of panels
    bar_bottom = pos_bottom.y0
    bar_height = pos_top.y1 - pos_bottom.y0
    bar_width = 0.016

    cax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, labelpad=2, fontsize=30)
    format_colorbar(cbar, tick_size=35, num_ticks=5, label_size=40)

# amplitude bar (right of amplitude column)
add_vertical_colorbar(
    ScalarMappable(norm=Normalize(vmin_amp, vmax_amp), cmap='inferno'),
    ax_amp_exp, ax_amp_ana, r'Power [dBm]', left_shift=0.01
)

# frequency bar (right of frequency column)
add_vertical_colorbar(
    ScalarMappable(norm=Normalize(vmin_freq, vmax_freq), cmap='inferno'),
    ax_freq_exp, ax_freq_ana, r'$\delta \omega_{\rm LC}/2\pi$ [MHz]',
    left_shift=0.01
)

# ── Cosmetics --------------------------------------------------------------
for ax in [ax_amp_exp, ax_amp_num, ax_amp_ana,
           ax_freq_exp, ax_freq_num, ax_freq_ana]:
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis='x', labelsize=35)
    ax.axvline(threshold, ls='--', lw=3, color='crimson')

for ax in [ax_freq_exp, ax_freq_num]:
    ax.set_yticks([]); ax.set_yticklabels([])

for ax in [ax_amp_exp, ax_amp_num]:
    ax.set_ylabel(r'$\phi$', fontsize=45)

# experiment / numerics / theory labels
for ax, lab in zip(
    [ax_amp_exp, ax_amp_num, ax_amp_ana,
     ax_freq_exp, ax_freq_num, ax_freq_ana],
    ['experiment', 'numerics', 'theory',
     'experiment', 'numerics', 'theory']
):
    ax.text(0.05, 0.95, lab, transform=ax.transAxes,
            fontsize=30, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='lightgrey', ec='none'))

# panel letters
for ax, letter in zip(
    [ax_amp_exp, ax_amp_num, ax_amp_ana],
    list('abcdef')
):
    ax.text(-0.15, 1.10, rf'$\textbf{{{letter}}}$',
            transform=ax.transAxes,
            fontsize=45, fontweight='bold', ha='right', va='top')

for ax, letter in zip(
    [ax_freq_exp, ax_freq_num, ax_freq_ana],
    list('def')
):
    ax.text(-0.05, 1.10, rf'$\textbf{{{letter}}}$',
            transform=ax.transAxes,
            fontsize=45, fontweight='bold', ha='right', va='top')

# ### Raster the contours so that it renders properly
for coll in c_amp_exp.collections:
    coll.set_rasterized(True)
for coll in c_amp_num.collections:
    coll.set_rasterized(True)
for coll in c_freq_exp.collections:
    coll.set_rasterized(True)
for coll in c_freq_num.collections:
    coll.set_rasterized(True)

plt.savefig('../plots/Fig_3.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_3.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_3.svg', bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
