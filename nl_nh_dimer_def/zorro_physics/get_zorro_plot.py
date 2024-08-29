import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.ticker as ticker

# Update matplotlib settings for consistent font and style
plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = True

# Define the epsilon value (could be parameterized or altered as needed)
epsilon_dBm = 4
omega_c = 6.027e9

# Load and process FFT data from directory
def load_and_prepare_data(directory):
    files = [f for f in os.listdir(directory) if f.startswith('fft_results_omega_d_') and f.endswith('_GHz.csv')]
    files.sort(key=lambda x: float(x.split('_')[4]))  # Sort files by drive frequency

    all_fft_magnitudes = []
    all_drive_frequencies = []
    fft_frequencies = None

    for file in files:
        drive_freq = float(file.split('_')[4]) * 1e9  # Extract and convert drive frequency to Hz

        all_drive_frequencies.append(drive_freq)
        df = pd.read_csv(os.path.join(directory, file))
        
        fft_frequencies = omega_c - df['FFT Frequency [Hz]'].values
        
        normalized_fft_magnitude = df['FFT Magnitude']
        all_fft_magnitudes.append(normalized_fft_magnitude)

    return fft_frequencies, np.array(all_drive_frequencies), np.array(all_fft_magnitudes)

# Plotting function for the 2D colorplot
def plot_colorplot(fft_frequencies, drive_frequencies, data_array):
    plt.figure(figsize=(9, 7))
    extent = [fft_frequencies.max()/1e9, fft_frequencies.min()/1e9, drive_frequencies.min()/1e9, drive_frequencies.max()/1e9]

    plt.imshow(data_array, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=-60)
    cbar = plt.colorbar(label='Amplitude [dBm]')

    ###############################################################################
    ## Same configuration as for experimental Fig. 4 in the paper for the colobar
    label_size = 30
    tick_size = 30
    num_ticks = 5

    cbar.ax.tick_params(labelsize=tick_size)
    tick_locator = ticker.MaxNLocator(nbins=num_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.label.set_size(label_size)
    ###############################################################################

    plt.xlabel(r'Meas. Freq. [GHz]')
    plt.ylabel(r'Drive Freq. [GHz]')

    plt.xlim((omega_c-8e6)/1e9, (omega_c+8e6)/1e9)
    plt.ylim((omega_c-8e6)/1e9, (omega_c+8e6)/1e9)

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.savefig(f'zorro_plot_{epsilon_dBm}_dBm.png', dpi=300)

# Specify the directory where your FFT data is stored
directory = f'data/epsilon_{epsilon_dBm}_dBm/freq/'
fft_frequencies, drive_frequencies, data_array = load_and_prepare_data(directory)

# Plot the colorplot
plot_colorplot(fft_frequencies, drive_frequencies, data_array)
