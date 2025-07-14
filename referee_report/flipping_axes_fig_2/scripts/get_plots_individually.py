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
threshold = 4.78

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

def find_nearest_index(array, value):

    index = np.abs(array - value).argmin()
    return index

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
    ax_bottom.plot(net_gains_t, max_transmissions_t_below, color='gray', ls='--', lw=2.0)
    ax_bottom.set_ylabel(r'$S_{21}^{\rm{max}}$ [dB]', fontsize=20)
    ax_bottom.tick_params(axis='y', labelsize=20)

    # Plotting max transmission above the center frequency
    ax_bottom.scatter(net_gains, max_transmissions_above, color='darkblue', label=r'Right Peak')
    ax_bottom.plot(net_gains_t, max_transmissions_t_above, color='gray', ls='--', lw=2.0, label=r'Numerics')

    # # Add the custom legend to the plot with the handles
    # ax_bottom.legend(handles=legend_handles, loc='best', fontsize=15)
    # ax_bottom.legend(loc='best', fontsize=10)

    ## so it doesn't look as crowded.
    h, l = ax_bottom.get_legend_handles_labels()   # Left, Right, Numerics

    # force the order you want (Left-Peak, Right-Peak, Numerics)
    order   = [0, 1, 2]
    handles = [h[i] for i in order]
    labels  = [l[i] for i in order]

    # 3 items + ncol=2  →  first row has 2 items, second row 1 item
    ax_bottom.legend(handles, labels,
                    ncol=2,            # << two columns
                    loc='best',
                    fontsize=12,
                    columnspacing=0.8, # tweak spacing as you like
                    handletextpad=0.4,
                    frameon=False)
    
    ax_bottom.set_xlim(-4.6, 8.4)
    ax_bottom.axvline(x=threshold, ls='--', lw=3.0, color='k', alpha=0.8)

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
    ax_bottom.set_ylabel(r'$S_{21}^{\rm{max}}$ [dB]', color='crimson', fontsize=20)
    ax_bottom.tick_params(axis='y', labelcolor='crimson', labelsize=20)

    index_closest = np.argmin(np.abs(np.array(max_transmissions_t)))

    # Get the corresponding net gain value
    closest_net_gain = net_gains_t[index_closest]

    print(f"The net gain value closest to a maximum transmission of 0.0 is: {closest_net_gain}")

    # phi_text = r'$\phi = \pi$'
    # ax_bottom.text(0.5, 1.02, phi_text, transform=ax_bottom.transAxes, fontsize=25, color='black',
    #         verticalalignment='bottom', horizontalalignment='center')

    # Create twin axis for FWHM on the bottom panel's axis
    ax_bottom2 = ax_bottom.twinx()

    # ## One referee requested looking at the log of the FWHM. I'm going to attach this in the referee report.
    # ax_bottom2.scatter(net_gains, np.log(fwhms), color='darkblue', label='FWHM')
    # ax_bottom2.plot(net_gains_t, np.log(fwhms_t), color='darkblue', ls='--', lw=2.0)

    ax_bottom2.scatter(net_gains, fwhms, color='darkblue', label='FWHM')
    ax_bottom2.plot(net_gains_t, fwhms_t, color='darkblue', ls='--', lw=2.0)

    ax_bottom2.set_ylabel(r'FWHM [MHz]', color='darkblue', fontsize=20)
    ax_bottom2.tick_params(axis='y', labelcolor='darkblue', labelsize=20)
    ax_bottom2.tick_params(axis='both', which='major', labelsize=20)
    ax_bottom2.set_xlim(-4.6, 8.4)
    ax_bottom2.axvline(x=threshold, ls='--', lw=3.0, color='k', alpha=0.8)