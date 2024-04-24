import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

import matplotlib.colors as mcolors

phases = ['hermitian', 'nonhermitian']
threshold_phase_diagram = np.mean([4.56, 4.48])  # with and without deltaphi

for phase in phases:
    phase_dir = f"images_phase_{phase}"
    os.makedirs(phase_dir, exist_ok=True)

    phase_dir_combined = f"images_phase_{phase}/combined_transmission"
    os.makedirs(phase_dir_combined, exist_ok=True)

    # Define the base paths for the experimental and numerical data
    data_path_experimental = f"experimental_data/{phase}"
    data_path_numerical = f"numerical_data/{phase}"

    # Use glob to find all relevant CSV files within the directories
    files_experimental = glob.glob(f"{data_path_experimental}/hermitian_phase_symmetric_attenuation_*.csv")
    files_numerical = glob.glob(f"{data_path_numerical}/hermitian_phase_symmetric_attenuation_*.csv")

    # Create a common figure for the combined spectrum
    plt.figure(figsize=(8, 8))

    # Assuming the filenames are structured as '..._attenuation_X.csv' where X is the attenuation level
    attenuation_files = [float(file.split('_')[-2]) for file in files_experimental]
    attenuations = sorted(list(set(attenuation_files)))  # Sort and remove duplicates to get unique attenuation levels

    # Assuming 'attenuations' is a list of all unique attenuation values sorted in ascending order
    attenuation_norm = mcolors.Normalize(vmin=min(attenuations), vmax=max(attenuations))
    attenuation_cmap = plt.cm.inferno  # Choose the colormap you want to use

    # Process each experimental file and plot it with the corresponding numerical data
    for file_exp in files_experimental:
        attenuation = float(file_exp.split('_')[-2])

        ## Ignore the point where the transmission is way above 0dB for the Hermitian phase...
        if (attenuation > -5.5 or attenuation <= 9.5):
            df_exp = pd.read_csv(file_exp, header=None)
            fs_exp = df_exp[0].values / 1e9  # Convert to GHz
            transmission_exp = df_exp[1].values + 20  # Apply external attenuation

            # Find the corresponding numerical data file
            file_num = f"{data_path_numerical}/hermitian_phase_symmetric_attenuation_{attenuation}_dB.csv"
            if os.path.exists(file_num):
                df_num = pd.read_csv(file_num)
                fs_num = df_num['Frequency'] / 1e9
                transmission_num = df_num['Transmission']

                attenuation_color = attenuation_cmap(attenuation_norm(attenuation))

                # Plot experimental and numerical data on the common figure
                plt.plot(fs_exp, transmission_exp, color=attenuation_color, label=f'Exp Attenuation: {attenuation} dB')
                plt.plot(fs_num, transmission_num, '--', color=attenuation_color, lw=2.0, label=f'Num Attenuation: {attenuation} dB')

                # Plot individual experimental and numerical data in separate figures
                plt.figure()
                plt.plot(fs_exp, transmission_exp, 'o', label=f'Exp Attenuation: {attenuation} dB')
                plt.plot(fs_num, transmission_num, '--', lw=2.0, color='crimson', label=f'Num Attenuation: {attenuation} dB')
                plt.xlabel('Frequency [GHz]', fontsize=14)
                plt.ylabel(r'$S_{21}$ [dB]', fontsize=14)
                plt.title(f'Transmission Spectrum at {attenuation} dB')
                plt.legend()
                plt.savefig(f'{phase_dir_combined}/transmission_comparison_{attenuation}_phase_{phase}.png')
                plt.close()

    # Customize and save the combined spectrum plot
    plt.xlabel('Frequency [GHz]', fontsize=14)
    plt.ylabel(r'$S_{21}$ [dB]', fontsize=14)
    plt.title(f'Combined Transmission Spectrum for {phase.capitalize()} Phase')
    plt.tight_layout()
    plt.savefig(f'{phase_dir}/combined_transmission_spectrum_{phase}.png')
    plt.close()

# Function to generate a colormap subplot
def generate_colormap(data_path, plot_title, subplot_position, with_headers=True, frequency_range=None):
    files = glob.glob(f"{data_path}/*.csv")
    all_transmissions = []
    all_frequencies = []
    all_attenuations = []

    # Process each file and accumulate the data for the colormap
    for file in files:
        # Read the file based on whether it has headers
        if with_headers:
            df = pd.read_csv(file)
            frequency = df['Frequency'].values / 1e9  # Convert to GHz
            transmission = df['Transmission'].values
        else:
            df = pd.read_csv(file, header=None)
            frequency = df[0].values / 1e9  # Convert to GHz
            transmission = df[1].values + 20  # Apply external attenuation

        attenuation = float(file.split('_')[-2])#.replace('.csv', '').replace('dB', ''))
        
        # Apply frequency range limits if specified
        if frequency_range:
            mask = (frequency >= frequency_range[0]) & (frequency <= frequency_range[1])
            frequency = frequency[mask]
            transmission = transmission[mask]

        all_transmissions.append(transmission)
        all_frequencies.append(frequency)
        all_attenuations.append(attenuation)

    # Sort the data by attenuation
    sorted_indices = np.argsort(all_attenuations)
    all_transmissions = np.array(all_transmissions)[sorted_indices]
    all_frequencies = np.array(all_frequencies)[sorted_indices]
    all_attenuations = np.array(all_attenuations)[sorted_indices]

    # Ensure all frequency arrays are identical, you may need to interpolate if they are not
    freq_grid, att_grid = np.meshgrid(all_frequencies[0], all_attenuations)

    # Create the colormap subplot
    plt.subplot(subplot_position)

    if data_path == 'experimental_data/hermitian' or data_path == 'numerical_data/hermitian':
        plt.imshow(all_transmissions, aspect='auto', extent=[freq_grid.min(), freq_grid.max(), att_grid.min(), att_grid.max()],
                interpolation='nearest', cmap='inferno', origin='lower', vmax=0, vmin=-48)
    else:
        plt.imshow(all_transmissions, aspect='auto', extent=[freq_grid.min(), freq_grid.max(), att_grid.min(), att_grid.max()],
                interpolation='nearest', cmap='inferno', origin='lower', vmax=28)

    plt.axhline(threshold_phase_diagram, ls = '--', lw=2.0, color='white', alpha=0.5)

    plt.colorbar(label='Transmission [dB]')
    plt.title(plot_title)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Net Gain [dB]')

phases = ['hermitian', 'nonhermitian']

for phase in phases:
    phase_dir = f"images_phase_{phase}"
    os.makedirs(phase_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))

    # Define frequency range for Hermitian phase
    frequency_limits = (5.975, 6.09) if phase == 'hermitian' else None

    # Plot experimental data as the first subplot
    data_path_experimental = f"experimental_data/{phase}"
    generate_colormap(data_path_experimental, f'Experimental Data - {phase.capitalize()} Phase', 121, with_headers=False, frequency_range=frequency_limits)

    # Plot numerical data as the second subplot
    data_path_numerical = f"numerical_data/{phase}"
    generate_colormap(data_path_numerical, f'Numerical Data - {phase.capitalize()} Phase', 122, with_headers=True, frequency_range=frequency_limits)

    # Save the figure with both subplots
    # plt.suptitle(f'Combined Colormap for {phase.capitalize()} Phase')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for the suptitle
    plt.savefig(f'{phase_dir}/combined_colormap_{phase}.png')
    plt.close()
