import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Update matplotlib settings
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

external_attenation = 20

def get_net_gain_and_phase_from_filename(filename):
    # Extract the net gain and phase value from the filename
    basename = os.path.basename(filename)
    parts = basename.split('_')
    net_gain = None
    phase = None
    for i, part in enumerate(parts):
        if part == 'gain':
            net_gain = float(parts[i + 1].replace('.csv', ''))
        elif part == 'phase':
            phase = float(parts[i + 1])
    return net_gain, phase

def plot_transmission_vs_frequency(phase, min_freq, max_freq):
    # Set the directory path
    folder_path = f'data/experiment/{phase}/'
    csv_files = glob(os.path.join(folder_path, '*.csv'))
    
    # Read the CSV files and extract data
    data = []
    net_gains = []
    phases = []
    
    for file in csv_files:
        net_gain, file_phase = get_net_gain_and_phase_from_filename(file)
        if net_gain is not None:
            net_gains.append(net_gain)
            phases.append(file_phase)
            df = pd.read_csv(file, header=None)
            frequencies = df.iloc[:, 0].values/1e9
            transmissions = df.iloc[:, 1].values + external_attenation
            
            # Apply frequency mask
            mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            filtered_frequencies = frequencies[mask]
            filtered_transmissions = transmissions[mask]
            
            if filtered_frequencies.size > 0:
                data.append((net_gain, filtered_frequencies, filtered_transmissions))
    
    if not data:
        print("No data in the specified frequency range.")
        return
    
    # Sort data by net gain in descending order for correct axis orientation
    data.sort(key=lambda x: x[0], reverse=True)
    net_gains_sorted = [d[0] for d in data][::-1]
    transmissions_sorted = np.array([d[2] for d in data])
    
    # Create 2D color plot
    plt.figure(figsize=(6, 6))
    plt.imshow(transmissions_sorted, aspect='auto', cmap='inferno', extent=[min_freq, max_freq, net_gains_sorted[0], net_gains_sorted[-1]])
    plt.colorbar(label='$S_{21}$ [dB]')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Net Gain [dB]')
    plt.tight_layout()
    plt.savefig(f'transmission_phase_{phase}.png')
    plt.close()

min_freq = 5.975
max_freq = 6.085
min_freq_single = min_freq + 0.025
max_freq_single = max_freq - 0.025

# Example of how to call the function with a specific frequency window
plot_transmission_vs_frequency('hermitian', min_freq, max_freq)
plot_transmission_vs_frequency('nonhermitian', min_freq_single, max_freq_single)
