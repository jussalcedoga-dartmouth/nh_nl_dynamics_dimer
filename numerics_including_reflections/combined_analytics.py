import matplotlib.pyplot as plt
from experiment_analytics import get_data
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.

def get_numerical_data(phase):
    # Path to the file where the results are saved
    file_path = f'numerical_data/{phase}_results.json'
    
    # Load the results from the file
    with open(file_path, 'r') as f:
        results = json.load(f)

    return results

def plot_hermitian_data(experimental_results, numerical_results, threshold):
    """ Plot data specific to hermitian phase. """
    exp_attenuations = sorted(experimental_results.keys())
    exp_max_below_freqs = [experimental_results[a]['max_below_freq'] for a in exp_attenuations if 'max_below_freq' in experimental_results[a]]
    exp_max_above_freqs = [experimental_results[a]['max_above_freq'] for a in exp_attenuations if 'max_above_freq' in experimental_results[a]]
    exp_max_trans_below = [experimental_results[a]['max_trans_below'] for a in exp_attenuations if 'max_trans_below' in experimental_results[a]]
    exp_max_trans_above = [experimental_results[a]['max_trans_above'] for a in exp_attenuations if 'max_trans_above' in experimental_results[a]]
    exp_distances = [experimental_results[a]['distance'] * 1e3 for a in exp_attenuations if 'distance' in experimental_results[a]]  # GHz to MHz

    num_attenuations = numerical_results['attenuations']
    max_transmission_left = numerical_results['max_transmission_left']
    max_transmission_right = numerical_results['max_transmission_right']
    num_left_peak_freqs = numerical_results['left_peak_freqs']
    num_right_peak_freqs = numerical_results['right_peak_freqs']
    num_peak_distances = numerical_results['peak_distances']

    print(num_attenuations)

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.scatter(exp_attenuations, exp_max_below_freqs, color='red', label='Exp Left Peak Frequency')
    plt.scatter(exp_attenuations, exp_max_above_freqs, color='blue', label='Exp Right Peak Frequency')
    plt.plot(num_attenuations, num_left_peak_freqs, 'r--', label='Num Left Peak Frequency')
    plt.plot(num_attenuations, num_right_peak_freqs, 'b--', label='Num Right Peak Frequency')
    plt.axvline(threshold, ls = '--', lw=2.0, color='black', label='Overall Gain Threshold')
    plt.axvline(0, ls = '--', lw=2.0, color='black', label='Hopping Gain Threshold')
    plt.title('Hermitian Phase: Peak Frequencies')
    plt.xlabel('Net Gain [dB]')
    plt.ylabel('Frequency [GHz]')
    plt.legend()

    plt.subplot(132)
    plt.scatter(exp_attenuations, exp_distances, color='royalblue', label='Exp Peak Distance')
    plt.plot(num_attenuations, num_peak_distances, ls = '--', lw = 2.0, color='crimson', label='Num Peak Distance')
    plt.axvline(threshold, ls = '--', lw=2.0, color='black', label='Overall Gain Threshold')
    plt.axvline(0, ls = '--', lw=2.0, color='black', label='Hopping Gain Threshold')
    plt.title('Hermitian Phase: Peak Distance')
    plt.xlabel('Net Gain [dB]')
    plt.ylabel('Distance [MHz]')
    plt.legend()

    plt.subplot(133)
    plt.scatter(exp_attenuations, exp_max_trans_below, color='red', label='Exp Max Transmission Left Peak')
    plt.scatter(exp_attenuations, exp_max_trans_above, color='blue', label='Exp Max Transmission Right Peak')
    plt.plot(num_attenuations, max_transmission_left, 'r--', label='Num Max Transmission Left')
    plt.plot(num_attenuations, max_transmission_right, 'b--', label='Num Max Transmission Right')
    plt.axvline(threshold, ls = '--', lw=2.0, color='black', label='Overall Gain Threshold')
    plt.axvline(0, ls = '--', lw=2.0, color='black', label='Hopping Gain Threshold')
    plt.title('Hermitian Phase: Max Transmission for Peaks')
    plt.xlabel('Net Gain [dB]')
    plt.ylabel('Transmission [dB]')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/hermitian_combined_data_plots.png')
    plt.close()

def plot_non_hermitian_data(experimental_results, numerical_results, threshold):
    """ Plot maximum transmission for non-hermitian phase in separate subfigures. """
    exp_attenuations = sorted(experimental_results.keys())
    exp_max_freqs = [experimental_results[a]['max_freq'] for a in exp_attenuations]
    exp_max_trans = [experimental_results[a]['max_trans'] for a in exp_attenuations]

    num_attenuations = numerical_results['attenuations']
    num_max_transmission_freqs = numerical_results['max_transmission_freqs']
    num_max_transmission_values = numerical_results['max_transmission_values']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))  # Create two subplots side by side

    # First subplot for frequency of maximum transmission
    ax1.scatter(exp_attenuations, exp_max_freqs, color='red', label='Exp Frequency of Max Transmission')
    ax1.plot(num_attenuations, num_max_transmission_freqs, 'r--', label='Num Frequency of Max Transmission')
    ax1.set_title('Non-Hermitian Phase: Frequency of Maximum Transmission')
    ax1.set_xlabel('Net Gain [dB]')
    ax1.set_ylim(5.98, 6.08)
    ax1.axvline(threshold, ls = '--', lw=2.0, color='black', label='Overall Gain Threshold')
    ax1.axvline(0, ls = '--', lw=2.0, color='black', label='Hopping Gain Threshold')
    ax1.set_ylabel('Frequency [GHz]')

    # Second subplot for maximum transmission value
    ax2.scatter(exp_attenuations, exp_max_trans, color='blue', label='Exp Maximum Transmission Value')
    ax2.plot(num_attenuations, num_max_transmission_values, 'b--', label='Num Maximum Transmission Value')
    ax2.axvline(threshold, ls = '--', lw=2.0, color='black', label='Overall Gain Threshold')
    ax2.axvline(0, ls = '--', lw=2.0, color='black', label='Hopping Gain Threshold')
    ax2.set_title('Non-Hermitian Phase: Maximum Transmission Value')
    ax2.set_xlabel('Net Gain [dB]')
    ax2.set_ylabel('Transmission [dB]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for suptitle
    plt.savefig('plots/non_hermitian_combined_max_transmission.png')
    plt.close()

def main():
    center_frequency = 6.02982  # Center frequency in GHz
    gain_threshold = np.mean([4.48, 4.56])  # Gain threshold in dB
    
    # Load experimental data
    hermitian_results = get_data("experimental_data/hermitian", center_frequency, is_hermitian=True)
    non_hermitian_results = get_data("experimental_data/nonhermitian", center_frequency, is_hermitian=False)

    # Load numerical data
    numerical_hermitian_results = get_numerical_data('hermitian')
    numerical_non_hermitian_results = get_numerical_data('nonhermitian')

    # Plot hermitian data
    plot_hermitian_data(hermitian_results, numerical_hermitian_results, gain_threshold)

    # Plot non-hermitian data
    plot_non_hermitian_data(non_hermitian_results, numerical_non_hermitian_results, gain_threshold)
if __name__ == '__main__':
    main()
