import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

def extract_info_from_filename(filename):
    # Parse filename to get the net gain
    parts = filename.split('_')
    net_gain = float(parts[4].replace('netgain', '').replace('.csv', ''))
    return net_gain

def plot_and_calculate_splitting(folder_path, center_freq=6.028):
    file_pattern = os.path.join(folder_path, 'phase_*_net_gain_*.csv')
    files = glob(file_pattern)
    
    results = []
    plot_folder = 'splitting_plots'
    os.makedirs(plot_folder, exist_ok=True)  # Ensure the directory for plots exists
    
    for file in files:
        net_gain = extract_info_from_filename(os.path.basename(file))
        df = pd.read_csv(file)
        external_attenuation = 20

        frequencies = df.iloc[:, 0].values / 1e9  # Convert from Hz to GHz
        transmissions = df.iloc[:, 1].values + external_attenuation

        mask_below = frequencies < center_freq
        mask_above = frequencies > center_freq

        if np.any(mask_below) and np.any(mask_above):
            max_index_below = np.argmax(transmissions[mask_below])
            max_index_above = np.argmax(transmissions[mask_above]) + np.sum(mask_below)

            freq_max_below = frequencies[mask_below][max_index_below]
            freq_max_above = frequencies[mask_above][max_index_above - np.sum(mask_below)]
            difference = np.abs(freq_max_below - freq_max_above) * 1e3

            results.append((net_gain, difference))

            # Plotting the individual trace with highlighted peaks
            plt.figure(figsize=(8, 6))
            plt.plot(frequencies, transmissions, label=r'$S_{21}$', lw=3.0)
            plt.scatter([freq_max_below, freq_max_above], [transmissions[mask_below][max_index_below], transmissions[mask_above][max_index_above - np.sum(mask_below)]], color='red', zorder=5, label='Max Peaks')
            plt.xlabel('Frequency [GHz]')
            plt.ylabel(r'$S_{21}$ [dB]')
            plt.title(f'Net Gain: {net_gain} dB')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f'trace_net_gain_{net_gain:.2f}_dB.png'))
            plt.close()

    # Sort results by net gain
    results.sort()
    net_gains, differences = zip(*results)  # Unzip the list of tuples

    plt.figure(figsize=(10, 6))
    plt.scatter(net_gains, differences)
    plt.xlabel('Net Gain [dB]')
    plt.ylabel('Splitting Distance [MHz]')
    plt.savefig('splitting_vs_net_gain.png')
    plt.close()

    kappa_cs = (np.array(differences)/2)/(10**(np.array(net_gains)/20))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(net_gains, kappa_cs)
    plt.xlabel('Net Gain [dB]')
    plt.ylabel('$\kappa_c$ [MHz]')
    plt.savefig('kappa_c_vs_net_gain.png')
    plt.close()

    print(f'kappa_c {np.mean(kappa_cs[-8:]):.3f} MHz, Mean of the last 8 points')

plot_and_calculate_splitting('data/experiment/hermitian')
