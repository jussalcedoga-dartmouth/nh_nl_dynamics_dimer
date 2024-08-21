import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.

#### Preamble
# Set the font globally to Helvetica
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

# Define paths to the directories containing the experimental and numerical data
script_directory = os.getcwd() # Adjusted for this environment, normally would be os.path.dirname(os.path.realpath(__file__))

# Now you can build paths relative to the script location
data_base_path = os.path.join(script_directory, '..', 'data')
experiment_path = os.path.join(data_base_path, 'experiment')
theory_path = os.path.join(data_base_path, 'theory')

# Define the structure of the directories and filenames
phases = ['hermitian', 'nonhermitian']
data_types = ['experiment', 'theory']

external_attenuation = 20

# Function to load data from a CSV file
def load_data(filepath, has_header):
    if has_header:
        data = pd.read_csv(filepath)
    else:
        data = pd.read_csv(filepath, header=None, names=['Frequency', 'Power_dB'])
        data['Power_dB'] += external_attenuation
    return data

# Function to extract net gain from the filename
def extract_net_gain(filename):
    gain_part = filename.split('_')[-1].replace('.csv', '')
    gain_value = float(gain_part)
    return gain_value

# # Function to create 2D colorplots using imshow
def create_colorplot(data, ax, title, xlabel, ylabel, data_type):

    net_gains = np.array([d[0] for d in data])

    transmissions = np.array([d[1]['Power_dB'].values for d in data])
    frequencies = data[0][1]['Frequency'].values/1e9

    # Apply the phase-specific settings
    if title == 'hermitian':
        freq_mask = (frequencies >= 5.975) & (frequencies <= 6.085)
        vmin, vmax = -39, -14
    elif title == 'nonhermitian':
        # freq_mask = (frequencies >= 6.0) & (frequencies <= 7.035)
        freq_mask = (frequencies >= 5.975) & (frequencies <= 6.085)
        vmin, vmax = -35, 28

    # Filter each transmission data set based on the frequency mask
    transmissions = np.array([t[freq_mask] for t in transmissions])
    frequencies = frequencies[freq_mask]

    if data_type == 'experiment':
        # img = ax.imshow(transmissions, aspect='auto', 
        #                 extent=[frequencies.min(), frequencies.max(), net_gains.min(), net_gains.max()],
        #                 origin='lower',  cmap='inferno', vmin=vmin, vmax=vmax)
        img = ax.imshow(transmissions, aspect='auto', 
                extent=[5.975, 6.085, net_gains.min(), net_gains.max()],
                origin='lower',  cmap='inferno', vmin=vmin, vmax=vmax)
    else:
        # img = ax.imshow(transmissions, aspect='auto', 
        #                 extent=[frequencies.min(), frequencies.max(), net_gains.min(), net_gains.max()],
        #                 origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        img = ax.imshow(transmissions, aspect='auto', 
                    extent=[5.975, 6.085, net_gains.min(), net_gains.max()],
                    origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    
    # # Set ticks for the x-axis
    ax.tick_params(axis='both', which='major', labelsize=22)

    text_label = 'experiment' if data_type == 'experiment' else 'theory'
    ax.text(0.1, 0.05, text_label, transform=ax.transAxes, fontsize=18, 
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    phi_text = r'$\phi = 0$' if title == 'hermitian' else r'$\phi = \pi$'
    ax.text(0.95, 0.05, phi_text, transform=ax.transAxes, fontsize=25, color='white',
            verticalalignment='bottom', horizontalalignment='right')

    # Set y-labels for ax1 and ax3
    if ax in [ax1, ax3]:
        ax.set_ylabel(ylabel, fontsize=25)

    # Set x-labels for ax3 and ax4
    if ax in [ax3, ax4]:
        ax.set_xlabel(xlabel, fontsize=25)

    if ax in [ax2, ax4]:  # These are the axes we want to add colorbars to
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='7%', pad=0.1)  # Adjust size and padding of colorbar
    
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=22)
        cbar.set_label(r'$S_{21}$ [dB]', size=22)

def find_fwhm(frequencies, transmissions):
    # Find the peak transmission and its index
    peak_transmission = np.max(transmissions)
    try:
        peak_index = np.argmax(transmissions)
        
        # Find the half maximum points
        half_max = peak_transmission - 3
        lower_index = np.where(transmissions[:peak_index] <= half_max)[0][-1]
        upper_index = np.where(transmissions[peak_index:] <= half_max)[0][0] + peak_index

        # Calculate FWHM
        fwhm = frequencies[upper_index] - frequencies[lower_index]
    except:
        fwhm = 0
    return fwhm

def create_bottom_panel_hermitian_phase(data, data_num, ax_bottom):

    center_freq=6.027e9

    ### Exp
    net_gains = np.array([d[0] for d in data])

    max_transmissions_below = [np.max(d[1][d[1]['Frequency'] < center_freq]['Power_dB'].values) for d in data]
    max_transmissions_above = [np.max(d[1][d[1]['Frequency'] > center_freq]['Power_dB'].values) for d in data]

    ### Theory
    net_gains_t = [d[0] for d in data_num]
    max_transmissions_t_below = [np.max(d[1][d[1]['Frequency'] < center_freq]['Power_dB'].values) for d in data_num]
    max_transmissions_t_above = [np.max(d[1][d[1]['Frequency'] > center_freq]['Power_dB'].values) for d in data_num]

    # ### Linewidhts
    ### Exp
    fwhms = [find_fwhm(d[1]['Frequency'].values, d[1]['Power_dB'].values) for d in data]
    fwhms = np.array(fwhms) / 1e6  # convert to MHz if needed

    ### Theory
    fwhms_t = [find_fwhm(d[1]['Frequency'].values, d[1]['Power_dB'].values) for d in data_num]
    fwhms_t = np.array(fwhms_t) / 1e6  # convert to MHz if needed

    print("\nExperimental FWHM (MHz):", fwhms)
    print("\nNumerical FWHM (MHz):", fwhms_t)

    #### For linewidth and separation #### #### #### #### #### #### #### #### #### 
    # Initialize lists to store maximum frequencies below and above the center frequency
    ### Experimental Data
    exp_freqs_below = []
    exp_freqs_above = []
    for d in data:
        df_below = d[1][d[1]['Frequency'] < center_freq]
        df_above = d[1][d[1]['Frequency'] > center_freq]
        if not df_below.empty:
            max_freq_below = df_below.loc[df_below['Power_dB'].idxmax(), 'Frequency']
            exp_freqs_below.append(max_freq_below)
        if not df_above.empty:
            max_freq_above = df_above.loc[df_above['Power_dB'].idxmax(), 'Frequency']
            exp_freqs_above.append(max_freq_above)

    ### Numerical Data
    num_freqs_below = []
    num_freqs_above = []
    for d in data_num:
        df_below = d[1][d[1]['Frequency'] < center_freq]
        df_above = d[1][d[1]['Frequency'] > center_freq]
        if not df_below.empty:
            max_freq_below = df_below.loc[df_below['Power_dB'].idxmax(), 'Frequency']
            num_freqs_below.append(max_freq_below)
        if not df_above.empty:
            max_freq_above = df_above.loc[df_above['Power_dB'].idxmax(), 'Frequency']
            num_freqs_above.append(max_freq_above)

    # Calculate the differences between peak frequencies below and above the center frequency
    exp_peak_differences = [(above - below) / 1e6 for below, above in zip(exp_freqs_below, exp_freqs_above)]  # in MHz
    num_peak_differences = [(above - below) / 1e6 for below, above in zip(num_freqs_below, num_freqs_above)]  # in MHz

    # Print the results
    print("\nExperimental Peak Differences (MHz):", exp_peak_differences)
    print("\nNumerical Peak Differences (MHz):", num_peak_differences)
    #### For linewidth and separation #### #### #### #### #### #### #### #### ####  
    
    # Plotting max transmission below the center frequency
    ax_bottom.scatter(net_gains, max_transmissions_below, color='crimson', label=r'Left Peak')
    ax_bottom.plot(net_gains_t, max_transmissions_t_below, color='crimson', ls='--', lw=2.0)
    ax_bottom.set_ylabel(r'$S_{21}^{\rm{max}}$ [dB]', fontsize=25)
    ax_bottom.tick_params(axis='y', labelsize=22)

    # Plotting max transmission above the center frequency
    ax_bottom.scatter(net_gains, max_transmissions_above, color='darkblue', label=r'Right Peak')
    ax_bottom.plot(net_gains_t, max_transmissions_t_above, color='darkblue', ls='--', lw=2.0)
    # Create custom legend handles
    legend_handles = [
        mlines.Line2D([], [], color='crimson', marker='o', linestyle='--', markersize=10, label='Left Peak'),
        mlines.Line2D([], [], color='darkblue', marker='o', linestyle='--', markersize=10, label='Right Peak')
    ]

    # Add the custom legend to the plot with the handles
    ax_bottom.legend(handles=legend_handles, loc='best', fontsize=15)

    phi_text = r'$\phi = 0$'
    ax_bottom.text(0.5, 1.02, phi_text, transform=ax_bottom.transAxes, fontsize=25, color='black',
            verticalalignment='bottom', horizontalalignment='center')

# Function to create the bottom panel
def create_bottom_panel(data, data_num, ax_bottom):
    ### Exp
    net_gains = np.array([d[0] for d in data])

    max_transmissions = [np.max(d[1]['Power_dB'].values) for d in data]
    fwhms = [find_fwhm(d[1]['Frequency'].values, d[1]['Power_dB'].values) for d in data]
    fwhms = np.array(fwhms) / 1e6  # convert to MHz if needed

    ### Theory
    net_gains_t = [d[0] for d in data_num]
    max_transmissions_t = [np.max(d[1]['Power_dB'].values) for d in data_num]
    fwhms_t = [find_fwhm(d[1]['Frequency'].values, d[1]['Power_dB'].values) for d in data_num]
    fwhms_t = np.array(fwhms_t) / 1e6  # convert to MHz if needed

    # Plotting max transmission on the bottom panel's axis
    ax_bottom.scatter(net_gains, max_transmissions, color='crimson', label=r'$S_{21}^{\rm{max}} \ \rm{[dB]}$')
    ax_bottom.plot(net_gains_t, max_transmissions_t, color='crimson', ls='--', lw=2.0)

    # ax_bottom.set_xlabel(r'Net Gain $\Delta G$ [dB]', fontsize=22)
    ax_bottom.set_ylabel(r'$S_{21}^{\rm{max}}$ [dB]', color='crimson', fontsize=22)
    ax_bottom.tick_params(axis='y', labelcolor='crimson', labelsize=22)

    index_closest = np.argmin(np.abs(np.array(max_transmissions_t)))

    # Get the corresponding net gain value
    closest_net_gain = net_gains_t[index_closest]

    print(f"The net gain value closest to a maximum transmission of 0.0 is: {closest_net_gain}")

    phi_text = r'$\phi = \pi$'
    ax_bottom.text(0.5, 1.02, phi_text, transform=ax_bottom.transAxes, fontsize=25, color='black',
            verticalalignment='bottom', horizontalalignment='center')

    # Create twin axis for FWHM on the bottom panel's axis
    ax_bottom2 = ax_bottom.twinx()
    ax_bottom2.scatter(net_gains, fwhms, color='darkblue', label='FWHM')
    ax_bottom2.plot(net_gains_t, fwhms_t, color='darkblue', ls='--', lw=2.0)
    ax_bottom2.set_ylabel(r'FWHM [MHz]', color='darkblue', fontsize=22)
    ax_bottom2.tick_params(axis='y', labelcolor='darkblue', labelsize=25)
    ax_bottom2.tick_params(axis='both', which='major', labelsize=25)

# Load and sort the data by net gain
data_by_phase_type = {}
for phase in phases:
    data_by_phase_type[phase] = {}
    for data_type in data_types:
        path = experiment_path if data_type == 'experiment' else theory_path
        phase_path = os.path.join(path, phase)
        files = os.listdir(phase_path)
        sorted_files = sorted(files, key=extract_net_gain)

        # Load the data for each file
        data = [(extract_net_gain(f), load_data(os.path.join(phase_path, f), has_header=(data_type == 'theory')))
                for f in sorted_files]
        data_by_phase_type[phase][data_type] = data

fig = plt.figure(figsize=(9, 19))  # Increased figure size to maintain aspect ratio after cropping
gs = gridspec.GridSpec(6, 2, height_ratios=[1.2, 1.2, 0.05, 0.45, 0.45, 0.45], width_ratios=[1.0, 1.08])

gs.update(wspace=0.25, hspace=0.35)

# Hermitian plots
ax1 = fig.add_subplot(gs[0, 0])  # Hermitian experiment
ax2 = fig.add_subplot(gs[0, 1])  # Hermitian theory

# Nonhermitian plots
ax3 = fig.add_subplot(gs[1, 0])  # Nonhermitian experiment
ax4 = fig.add_subplot(gs[1, 1])  # Nonhermitian theory

ax_ghost = fig.add_subplot(gs[2, :])  # This subplot spans both columns
ax_ghost.axis('off')  # Turn off the ghost subplot's axes

# Bottom panel max transmission left and right peak Hermitian Phase (theory and experiment)
ax5 = fig.add_subplot(gs[3, :])  # Span the bottom panel across both columns

# Bottom panel max transmission nonHermitian phase (theory and experiment)
ax6 = fig.add_subplot(gs[4, :])  # Span the bottom panel across both columns

# Adjust the position of the bottom plot to prevent overlap
plt.subplots_adjust(bottom=0.1)

# Plotting the experimental and numerical data side by side for both phases
create_colorplot(data_by_phase_type['hermitian']['experiment'], ax1, 'hermitian', r'Frequency [GHz]', r'Net Gain, $\Delta G$ [dB]', 'experiment')
create_colorplot(data_by_phase_type['hermitian']['theory'], ax2, 'hermitian', '', '', 'theory')
create_colorplot(data_by_phase_type['nonhermitian']['experiment'], ax3, 'nonhermitian', r'Frequency [GHz]', r'Net Gain, $\Delta G$ [dB]', 'experiment')
create_colorplot(data_by_phase_type['nonhermitian']['theory'], ax4, 'nonhermitian', r'Frequency [GHz]', '', 'theory')

# # Now, call your function to fill in the bottom panel, passing in the Axes object
create_bottom_panel_hermitian_phase(data_by_phase_type['hermitian']['experiment'], data_by_phase_type['hermitian']['theory'], ax5)
create_bottom_panel(data_by_phase_type['nonhermitian']['experiment'], data_by_phase_type['nonhermitian']['theory'], ax6)

## Hermitian
for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))

## Non-Hermitian
for ax in [ax3, ax4]:
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))

ax6.set_xlabel(r'Net Gain, $\Delta G$ [dB]', fontsize=25)

# print(data_dissipation_rates.head())
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}",  # To enable bold math with \bm{}
})

helvetica_font = fm.FontProperties(family='Helvetica', weight='bold')

# Add labels outside the subfigure in the white space, near the top left of each subplot
ax1.text(-0.15, 1.05, r'$\textbf{a}$', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')
ax2.text(-0.1, 1.05, r'$\textbf{b}$', transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')
ax3.text(-0.15, 1.05, r'$\textbf{c}$', transform=ax3.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')
ax4.text(-0.1, 1.05, r'$\textbf{d}$', transform=ax4.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')
ax5.text(-0.15, 1.25, r'$\textbf{e}$', transform=ax5.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')
ax6.text(-0.15, 1.25, r'$\textbf{f}$', transform=ax6.transAxes, fontsize=30, fontweight='bold', va='top', ha='right')

# # Positioning the "Region I" text to the left of the vertical line
# ax5.text(0.48, 0.90, 'Region I', transform=ax5.transAxes, fontsize=25, color='black',
#          verticalalignment='top', horizontalalignment='center')

# # Positioning the "Region II" text to the right of the vertical line
# ax5.text(0.75, 0.90, 'Region II', transform=ax5.transAxes, fontsize=25, color='black',
#          verticalalignment='top', horizontalalignment='center')

# Save the figure with minimal white space
plt.savefig('../plots/Fig_2.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_2.pdf', bbox_inches='tight', pad_inches=0.1, dpi=400)
plt.savefig('../plots/Fig_2.svg', bbox_inches='tight', pad_inches=0.1, dpi=400)

plt.close()
