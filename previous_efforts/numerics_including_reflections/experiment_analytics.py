import os
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.


def load_data(file_path):
    external_attenuation = 20
    df = pd.read_csv(file_path, header=None)
    frequencies = df[0].values / 1e9  # Convert Hz to GHz
    transmission = df[1].values + external_attenuation
    return frequencies, transmission

def process_data(frequencies, transmission, center_frequency, is_hermitian=True):
    mask = (frequencies >= center_frequency - 0.105) & (frequencies <= center_frequency + 0.105)
    f_filtered = frequencies[mask]
    t_filtered = transmission[mask]

    max_trans_total = np.max(t_filtered)
    max_index = np.argmax(t_filtered)
    max_freq = f_filtered[max_index]

    if is_hermitian:
        below_omega_mask = f_filtered < center_frequency
        above_omega_mask = f_filtered > center_frequency

        if np.any(below_omega_mask) and np.any(above_omega_mask):
            max_below_freq = f_filtered[below_omega_mask][np.argmax(t_filtered[below_omega_mask])]
            max_above_freq = f_filtered[above_omega_mask][np.argmax(t_filtered[above_omega_mask])]
            max_trans_below = np.max(t_filtered[below_omega_mask])
            max_trans_above = np.max(t_filtered[above_omega_mask])
            distance = np.abs(max_below_freq - max_above_freq)

            return max_below_freq, max_above_freq, max_trans_below, max_trans_above, distance
        return None, None, None, None, None
    else:
        return max_freq, max_trans_total

def get_data(data_path, center_frequency, is_hermitian=True):
    results = {}
    files = glob.glob(f"{data_path}/*.csv")
    for file in files:
        frequencies, transmission = load_data(file)
        data = process_data(frequencies, transmission, center_frequency, is_hermitian)
        attenuation_level = float(file.split('_')[-2])
        if is_hermitian:
            if data[0]:
                results[attenuation_level] = {
                    'max_below_freq': data[0], 'max_above_freq': data[1],
                    'max_trans_below': data[2], 'max_trans_above': data[3],
                    'distance': data[4]
                }
        else:
            results[attenuation_level] = {
                'max_freq': data[0], 'max_trans': data[1]
            }
    return results
