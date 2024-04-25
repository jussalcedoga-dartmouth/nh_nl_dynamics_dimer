import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)

    ## Only consider data where the gain is larger than 3.5dB
    data = data[data['Gain [dB]'] >= 3.5]

    gains = np.sort(data['Gain [dB]'].unique())
    phases = np.sort(data['Phase [rad]'].unique())
    
    amplitude_matrix = np.zeros((len(phases), len(gains)))
    frequency_matrix = np.zeros((len(phases), len(gains)))

    for _, row in data.iterrows():
        gain_index = np.where(gains == row['Gain [dB]'])[0][0]
        phase_index = np.where(phases == row['Phase [rad]'])[0][0]
        amplitude_matrix[phase_index, gain_index] = row['Amplitude [dBm]']
        ## normalize the frequency to GHz
        frequency_matrix[phase_index, gain_index] = row['Frequency [Hz]']/1e9

    ## ignore the frequency where the amplitude is less than -40 dBm
    ## the maximum in such case is random (as there is no limit cycle ;)
    frequency_matrix[amplitude_matrix < -40] = np.nan

    return gains, phases, amplitude_matrix, frequency_matrix

def plot_data(gains, phases, amplitude, frequency, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Amplitude plot
    amp_contour = ax1.contourf(gains, phases, amplitude, levels=100, cmap='inferno')
    fig.colorbar(amp_contour, ax=ax1, label='Amplitude [dBm]')
    ax1.set_xlabel('Gain [dB]')
    ax1.set_ylabel('Phase [rad]')
    ax1.axvline(4.52, ls='--', lw=2.0, color='crimson')
    
    # Frequency plot
    freq_contour = ax2.contourf(gains, phases, frequency, levels=100, cmap='viridis')
    fig.colorbar(freq_contour, ax=ax2, label='Frequency [GHz]')
    ax2.set_xlabel('Gain [dB]')
    ax2.set_ylabel('Phase [rad]')
    ax2.axvline(4.52, ls='--', lw=2.0, color='crimson')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

data_base_path = '../data/experiment'
delta_phi_path = os.path.join(data_base_path, 'delta_phi', 'gain_phase_amplitude_frequency_deltaphi.csv')
no_delta_phi_path = os.path.join(data_base_path, 'no_delta_phi', 'gain_phase_amplitude_frequency_no_deltaphi.csv')

## plot data with and without delta phi
gains, phases, amplitude, frequency = load_data_from_csv(delta_phi_path)
plot_data(gains, phases, amplitude, frequency, 'With Delta Phi', '../plots/amplitude_frequency_with_delta_phi.png')

gains, phases, amplitude, frequency = load_data_from_csv(no_delta_phi_path)
plot_data(gains, phases, amplitude, frequency, 'Without Delta Phi', '../plots/amplitude_frequency_without_delta_phi.png')
