import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
# Update matplotlib settings for consistent font and style
plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = True

omega_c = 6.027e9

def load_and_process_data(data_folder_time, data_folder_freq):
    # List all files in the frequency domain directory
    filenames = [f for f in os.listdir(data_folder_freq) if f.startswith('fft_results_omega_d_') and f.endswith('_GHz.csv')]
    filenames.sort()

    # Process each frequency file alongside its corresponding time trace file
    for filename in filenames:
        # Extract drive frequency from filename
        drive_freq = float(filename.split('_')[4])
        
        # Load frequency domain data
        df_freq = pd.read_csv(os.path.join(data_folder_freq, filename))
        fft_frequencies = df_freq['FFT Frequency [Hz]']
        # fft_magnitude = df_freq['FFT Magnitude'] / df_freq['FFT Magnitude'].max()
        fft_magnitude = df_freq['FFT Magnitude']

        # Load time domain data
        df_time = pd.read_csv(os.path.join(data_folder_time, filename))
        time = df_time['time']
        alpha_2_real = df_time['alpha_2_real']
        alpha_2_imag = df_time['alpha_2_imag']

        # Plotting both time and frequency data side by side
        plt.figure(figsize=(16, 7))

        # Time domain plot
        plt.subplot(1, 2, 1)
        plt.plot(time*1e6, alpha_2_real, color='blue', label=r'Re[$\alpha_2$]')
        plt.plot(time*1e6, alpha_2_imag, color='crimson', label=r'Im[$\alpha_2$]')
        plt.xlabel(r'Time [$\mu$ s]')
        plt.ylabel(r'$\alpha_2$')
        plt.legend()
        plt.title(f'Time Trace at $\omega_d$: {drive_freq:.4f} GHz')

        # Frequency domain plot
        plt.subplot(1, 2, 2)
        plt.plot(fft_frequencies / 1e6, fft_magnitude, color='crimson')
        print((omega_c - drive_freq*1e9)*1e3/1e9)
        # plt.axvline((drive_freq*1e9 - omega_c)*1e3/1e9, ls='--', lw=2.0, color='k')
        plt.xlabel(r'$\delta \omega$ [MHz]')
        plt.ylabel('Amplitude [dBm]')
        plt.xlim(-8, 8)
        plt.title(rf'FFT at $\omega_d$: {drive_freq:.4f} GHz')

        # Save the combined plot
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/combined_FFT_time_trace_{drive_freq:.4f}_GHz.png')
        plt.close()

epsilon_dBm = 4

# Directories for time and frequency domain data
data_folder_time = f'data/epsilon_{epsilon_dBm}_dBm/time'
data_folder_freq = f'data/epsilon_{epsilon_dBm}_dBm/freq'

# Directory for output plots
plots_dir = f'plots/FFT_plots_{epsilon_dBm}_dBm'
os.makedirs(plots_dir, exist_ok=True)

# Process the data and generate plots
load_and_process_data(data_folder_time, data_folder_freq)
