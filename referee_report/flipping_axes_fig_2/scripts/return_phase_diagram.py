import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
import matplotlib.font_manager as fm
# Create custom colormap: yellow inside the contour, black outside
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['black', 'yellow'])

plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Light'], 'text.usetex': True})

# Define custom formatter to handle specific label formatting
def set_yaxis_ticks(ax, tick_size=35):
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
    
def format_colorbar(cbar, tick_size=30, num_ticks=5, label_size=30):
    cbar.ax.tick_params(labelsize=tick_size)
    # Set the number of ticks in the colorbar
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)

def format_colorbar_freq(cbar, tick_size=30, num_ticks=4, label_size=30):
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
    
    kappa_int = np.mean([kappa_int_1, kappa_int_2])
    kappa_in_out = np.mean([kappa_drive, kappa_readout])

    num = (10**(gain/20) * (1 + np.sin(phase/2)) * (kappa_c**2 * omega1 * h_bar * alpha_sat + kappa_c * b_amp)) - 2*b_amp*(kappa_c + kappa_in_out + kappa_int)
    den = 2*kappa_c*omega1*h_bar*(kappa_c + kappa_in_out + kappa_int)
    N2 = num/den
    return N2

def create_amplitude_plot(ax):
    gain_values = np.linspace(-4.6, 8.4, 1000)
    phase_values = np.linspace(0, 2*np.pi, 1000)  # Phase from 0 to 2*pi

    gain_grid, phase_grid = np.meshgrid(gain_values, phase_values)
    amplitude_grid = amplitude_LC(gain_grid, phase_grid)

    ### Constant grid with reduced amplitude to show instability onset
    max_amplitude = np.max(amplitude_grid)
    constant_amplitude_grid = np.full_like(amplitude_grid, max_amplitude) - 10

    ### Analytic Amplitude of the Limit Cycle
    ## baseline experimental data
    power_dBm = -42
    power_watts = 10**(power_dBm / 10) * 1e-3

    # Calculating photon numbers
    photon_numbers_base = power_watts / (kappa_readout * h_bar * omega2)

    ### Using a constant amplitude to show the onset of instability
    masked_amplitude_grid = np.where(amplitude_grid > alpha_sat, constant_amplitude_grid, photon_numbers_base)

    # Calculate power in dBm
    photon_numbers = masked_amplitude_grid
    power_watts = photon_numbers * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)

    # Create binary mask for contour fill
    mask = amplitude_grid > alpha_sat

    # Plotting the masked region with yellow
    cf = ax.contourf(gain_grid, phase_grid, mask, levels=[0, 0.5, 1], cmap=custom_cmap)

    # Overlay a contour to indicate regions where amplitude exceeds alpha_sat
    contour_levels = [alpha_sat]
    contours = ax.contour(gain_grid, phase_grid, amplitude_grid, levels=contour_levels, colors='white', linewidths=3.0)

    threshold = 4.7822

    ax.axvline(x=threshold, ls='--', lw=3.0, color='white', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_ylabel(r'$\phi$', fontsize=30)
    set_yaxis_ticks(ax)
    ax.xaxis.set_major_locator(MaxNLocator(4))

    # ax.text(0.05, 0.8, r'$\textbf{Stable}$', transform=ax.transAxes, fontsize=30, color='white',
    ax.text(0.05, 0.7, r'stable', transform=ax.transAxes, fontsize=20, color='k',
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # ax.text(0.35, 0.45, r'$\textbf{Unstable/Limit Cycle}$', transform=ax.transAxes, fontsize=30, color='k',
    ax.text(
        0.805, 0.32,
        r'unstable' '\n' r'limit cycle',   # 2-line, LaTeX-rendered
        transform=ax.transAxes,
        fontsize=20, color='k',
        va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8)
    )

    return ax
