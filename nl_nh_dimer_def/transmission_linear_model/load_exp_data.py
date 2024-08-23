import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

external_attenuation = 20

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

def transmission_vs_frequency(phase, min_freq, max_freq):
    # Set the directory path
    folder_path = f'experiment_analysis/data/experiment/{phase}/'
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
            transmissions = df.iloc[:, 1].values + external_attenuation
            
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

    return transmissions_sorted, net_gains_sorted

def extract_info_from_filename(filename):
    # Parse filename to get the net gain
    parts = filename.split('_')
    net_gain = float(parts[4].replace('netgain', '').replace('.csv', ''))
    return net_gain

def get_metrics_spectrum(folder_path, center_freq=6.028):

    file_pattern = os.path.join(folder_path, 'phase_*_net_gain_*.csv')
    files = glob(file_pattern)
    results = []

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
            max_value_below = transmissions[max_index_below]
            max_value_above = transmissions[max_index_above]

            results.append((net_gain, difference, max_value_below, max_value_above))

    # Sort results by net gain
    results.sort()
    net_gains, differences, max_values_below, max_values_above = zip(*results)  # Unzip the list of tuples
    return net_gains, differences, max_values_below, max_values_above
    
def get_metrics_spectrum_nonhermitian(folder_path):

    file_pattern = os.path.join(folder_path, 'phase_*_net_gain_*.csv')
    files = glob(file_pattern)
    results = []

    for file in files:
        net_gain = extract_info_from_filename(os.path.basename(file))
        df = pd.read_csv(file)
        external_attenuation = 20

        transmissions = df.iloc[:, 1].values + external_attenuation

        max_index = np.argmax(transmissions)
        max_transmission = transmissions[max_index]
        results.append((net_gain, max_transmission))

    # Sort results by net gain
    results.sort()
    net_gains, max_transmissions = zip(*results)  # Unzip the list of tuples
    return net_gains, max_transmissions
