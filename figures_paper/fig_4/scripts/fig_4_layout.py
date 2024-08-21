import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

# plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = True

threshold = 4.78

fig = plt.figure(figsize=(21, 9))

# fig = plt.figure(figsize=(18, 9))
# Define GridSpec with a 'ghost' column between the large panel and the smaller plots
gs = gridspec.GridSpec(2, 4, width_ratios=[2.5, 0.2, 1.2, 1.4])  # Includes two 'ghost' columns for spacing

# Large panel on the left
ax_large_panel = fig.add_subplot(gs[:, 0])

ax_ghost = fig.add_subplot(gs[:, 1])
ax_ghost.axis('off')  # Turn off the ghost subplot's axes

# Subplots for forward and backward data in the middle columns
ax_forward_0 = fig.add_subplot(gs[0, 2])  # Middle-left column for LO Power: 0 dBm

ax_backward_0 = fig.add_subplot(gs[1, 2])  # Middle-left column for LO Power: 0 dBm

ax_forward_8 = fig.add_subplot(gs[0, 3])  # Middle-right column for LO Power: 8 dBm

ax_backward_8 = fig.add_subplot(gs[1, 3])  # Middle-right column for LO Power: 8 dBm

epsilons = ['None', '0', '4', '8', '12', '16']
colors = cm.inferno_r(np.linspace(0, 1, len(epsilons)))

for idx, epsilon in enumerate(epsilons):
    csv_filename = f'../data/experiment/contours/combs_characterization_epsilon_{epsilon}.csv'
    
    # Load data from CSV
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    phases = data[:, 0]
    
    # Read the attenuation headers correctly
    with open(csv_filename, 'r') as file:
        headers = file.readline().strip().split(',')
        attenuations = np.array([float(h) for h in headers[1:]])

    imshow_data = data[:, 1:]

    # Check dimensions
    if len(attenuations) != imshow_data.shape[1]:
        raise ValueError(f"Mismatch in the number of attenuations ({len(attenuations)}) and the number of columns in imshow_data ({imshow_data.shape[1]})")

    # Plotting on the left panel
    contour = ax_large_panel.contourf(attenuations, phases, imshow_data, levels=[0.5, 1], colors=[colors[idx]], alpha=0.6, origin='lower')

# Custom legend for the left panel
# legend_handles = [Patch(facecolor=colors[i], edgecolor='none', label=f'{epsilon} dBm') for i, epsilon in enumerate(epsilons) if epsilon != 'None' else label = 'None']
legend_handles = [Patch(facecolor=colors[i], edgecolor='none', label=f'{epsilon} dBm' if epsilon != 'None' else 'None') for i, epsilon in enumerate(epsilons)]
ax_large_panel.legend(handles=legend_handles, title=r"Drive Power", loc='upper left', fontsize=17, title_fontsize=17)

# Define the function to load and plot the forward sweep data
def load_and_plot_forward(ax, phase, lo_power):
    csv_path = f'../data/experiment/zorro_plots/synchronization_data/{phase}_LO_power_{lo_power}_intensity_mesh.csv'
    df = pd.read_csv(csv_path)
    intensity_mesh = df.values
    freq_mesh = df.columns.astype(float)  # assuming the columns are frequency values in GHz

    im = ax.imshow(intensity_mesh, aspect='auto', origin='lower',
                   extent=[freq_mesh[0], freq_mesh[-1], freq_mesh[0], freq_mesh[-1]],
                   interpolation='nearest', cmap='inferno')
    return im

# Define the function to load and plot the backward sweep data
def load_and_plot_backward(ax, phase, lo_power):
    csv_path = f'../data/experiment/zorro_plots/synchronization_reversed_data/{phase}_LO_power_{lo_power}_intensity_mesh.csv'
    df = pd.read_csv(csv_path)
    intensity_mesh = df.values
    freq_mesh = df.columns.astype(float)  # assuming the columns are frequency values in GHz

    im = ax.imshow(intensity_mesh, aspect='auto', origin='lower',
                   extent=[freq_mesh[0], freq_mesh[-1], freq_mesh[0], freq_mesh[-1]],
                   interpolation='nearest', cmap='inferno')
    
    return im

# Create subplots for forward and backward data in the first column on the right
# ax_forward_0 = fig.add_subplot(gs[0, 1])
im_forward_0 = load_and_plot_forward(ax_forward_0, 'nonhermitian', '0')

# ax_backward_0 = fig.add_subplot(gs[1, 1])
im_backward_0 = load_and_plot_backward(ax_backward_0, 'nonhermitian', '0')

# Add colorbars for forward and backward plots
cbar_forward_0 = fig.colorbar(im_forward_0, ax=ax_forward_0)
cbar_backward_0 = fig.colorbar(im_backward_0, ax=ax_backward_0)

# Repeat the plotting for LO_power = 8 in the second column on the right
# ax_forward_8 = fig.add_subplot(gs[0, 2])
# im_forward_8 = load_and_plot_forward(ax_forward_8, 'nonhermitian', '8')
im_forward_8 = load_and_plot_forward(ax_forward_8, 'nonhermitian', '4')

# ax_backward_8 = fig.add_subplot(gs[1, 2])
# im_backward_8 = load_and_plot_backward(ax_backward_8, 'nonhermitian', '8')
im_backward_8 = load_and_plot_backward(ax_backward_8, 'nonhermitian', '4')

# Add colorbars for forward and backward plots
cbar_forward_8 = fig.colorbar(im_forward_8, ax=ax_forward_8)
cbar_backward_8 = fig.colorbar(im_backward_8, ax=ax_backward_8)

# Add legends to colorbars only for the rightmost plots
cbar_forward_8.ax.set_ylabel(r'Amplitude  [dBm]', fontsize=18)
cbar_backward_8.ax.set_ylabel(r'Amplitude [dBm]', fontsize=18)

for cbar in [cbar_forward_8, cbar_backward_8]:
    label_size = 25
    tick_size = 25
    num_ticks = 5
    cbar.ax.tick_params(labelsize=tick_size)
    # Set the number of ticks in the colorbar
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)

# Remove the middle colorbars
cbar_forward_0.remove()
cbar_backward_0.remove()

def set_yaxis_ticks(ax, tick_size=40):
    # Set major locator to MultipleLocator (pi)
    ax.set_ylim(0, 2*np.pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
    # Define custom formatter to handle specific label formatting
    def format_func(value, pos):
        if np.isclose(value, 0):
            return '0'
        elif np.isclose(value, np.pi):
            return r'$\pi$'  # Display 'π' instead of '1π'
        elif np.isclose(value, 2 * np.pi):
            return r'2$\pi$'  # Optionally handle the 2π case as well
        else:
            return f'{value/np.pi:.1g}$\pi$'  # General case

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # Increase tick size
    ax.tick_params(axis='y', which='major', labelsize=tick_size)

for ax in [ax_large_panel]:
    ax.set_xlabel(r'$\Delta G$ [dB]', fontsize=40)
    ax.set_ylabel(r'$\phi$', fontsize=45)
    ax.tick_params(axis='both', which='major', labelsize=35)
    set_yaxis_ticks(ax)
    ax.set_xlim(4.01, ax.get_xlim()[1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

for ax in [ax_backward_0, ax_backward_8]:
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    # ax.set_xlabel(r'Drive  Frequency  [GHz]', fontsize=25, labelpad=15)
    ax.set_xlabel(r'Measured  Frequency  [GHz]', fontsize=27, labelpad=15)
    ax.tick_params(axis='x', which='major', labelsize=25)

for ax in [ax_forward_0, ax_backward_0]:
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    # ax.set_ylabel(r'Measured Frequency [GHz]', fontsize=25, labelpad=15)
    ax.set_ylabel(r'Drive Frequency [GHz]', fontsize=27, labelpad=15)
    ax.tick_params(axis='y', which='major', labelsize=25)

for ax in [ax_forward_0, ax_forward_8]:
    ax.set_xticks([])
    ax.set_xticklabels([])

for ax in [ax_forward_8, ax_backward_8]:
    ax.set_yticks([])
    ax.set_yticklabels([])

# Little annotations...
# Add annotation for the forward and backward sweeps at 0 dBm
ax_forward_0.annotate(r'0 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))
ax_backward_0.annotate(r'0 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))

# Add annotation for the forward and backward sweeps at 8 dBm
# ax_forward_8.annotate(r'8 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))
# ax_backward_8.annotate(r'8 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))

ax_forward_8.annotate(r'4 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))
ax_backward_8.annotate(r'4 dBm', xy=(0.8, 0.1), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=25, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))

# Adjust the layout and spacing between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}",  # To enable bold math with \bm{}
})

helvetica_font = fm.FontProperties(family='Helvetica', weight='bold')

# Add labels to each subplot with Helvetica
ax_large_panel.text(-0.1, 1.15, r'$\textbf{a}$', transform=ax_large_panel.transAxes, fontsize=50, fontproperties=helvetica_font, va='top', ha='right')
ax_forward_0.text(-0.1, 1.15, r'$\textbf{b}$', transform=ax_forward_0.transAxes, fontsize=45, fontproperties=helvetica_font, va='top', ha='right')
ax_backward_0.text(-0.1, 1.15, r'$\textbf{c}$', transform=ax_backward_0.transAxes, fontsize=45, fontproperties=helvetica_font, va='top', ha='right')
ax_forward_8.text(-0.05, 1.15, r'$\textbf{d}$', transform=ax_forward_8.transAxes, fontsize=45, fontproperties=helvetica_font, va='top', ha='right')
ax_backward_8.text(-0.05, 1.15, r'$\textbf{e}$', transform=ax_backward_8.transAxes, fontsize=45, fontproperties=helvetica_font, va='top', ha='right')

### Add little arrows...
# Further adjusted arrows for subfigures b and d (forward sweep)
ax_forward_0.arrow(0.05, 0.10, 0.12, 0.12, head_width=0.025, head_length=0.035, linewidth=1.5, fc='red', ec='red', transform=ax_forward_0.transAxes)
ax_forward_8.arrow(0.05, 0.10, 0.12, 0.12, head_width=0.030, head_length=0.035, linewidth=1.5,fc='red', ec='red', transform=ax_forward_8.transAxes)

# Further adjusted arrows for subfigures c and e (backward sweep)
ax_backward_0.arrow(0.95, 0.10, -0.12, 0.12, head_width=0.025, head_length=0.035, linewidth=1.5, fc='red', ec='red', transform=ax_backward_0.transAxes)
ax_backward_8.arrow(0.95, 0.10, -0.12, 0.12, head_width=0.030, head_length=0.035, linewidth=1.5, fc='red', ec='red', transform=ax_backward_8.transAxes)

# Save the entire layout figure
plt.savefig('../plots/Fig_4.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_4.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_4.svg', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.close()
