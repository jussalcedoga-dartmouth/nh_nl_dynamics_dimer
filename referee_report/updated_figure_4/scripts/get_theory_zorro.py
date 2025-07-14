import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib

# Update matplotlib settings for consistent font and style
plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = True

def find_nearest_index(array, value):

    index = np.abs(array - value).argmin()
    return index

def desired_ticks(freq_array, desired_ticks_positions):

    indices = [find_nearest_index(freq_array, tick) for tick in desired_ticks_positions]
    closest_frequencies = [freq_array[idx] for idx in indices]

    return closest_frequencies

def load_and_prepare_data(directory):

    omega_c = 6.027e9  # Central frequency in Hz

    files = [f for f in os.listdir(directory) if f.startswith('fft_results_omega_d_') and f.endswith('_GHz.csv')]
    files.sort(key=lambda x: float(x.split('_')[4]))  # Sort files by drive frequency

    all_fft_magnitudes = []
    all_drive_frequencies = []
    fft_frequencies = None

    for file in files:
        drive_freq = float(file.split('_')[4]) * 1e9  # Extract and convert drive frequency to Hz
        df = pd.read_csv(os.path.join(directory, file))
        
        if fft_frequencies is None:
            fft_frequencies = omega_c - df['FFT Frequency [Hz]'].values
        
        frequencies = omega_c - df['FFT Frequency [Hz]'].values
        normalized_fft_magnitude = df['FFT Magnitude']
        all_drive_frequencies.append(drive_freq)
        all_fft_magnitudes.append(normalized_fft_magnitude)

    return frequencies / 1e9, np.array(all_drive_frequencies), np.array(all_fft_magnitudes)

## Updated more robust new rotation!!!
import numpy as np
import scipy.ndimage as ndimage

def rotate_data_within_range(
    fft_frequencies,
    data_array,
    center_freq,
    bandwidth=20e-3,
    shift_scale=1.15, ## this looks good.
    wrap=True
):
    num_rows, num_cols = data_array.shape
    rotated_array = np.copy(data_array)
    
    # Identify the subset of columns within [center_freq - bandwidth, center_freq + bandwidth].
    freq_min, freq_max = center_freq - bandwidth, center_freq + bandwidth
    indices = np.where((fft_frequencies >= freq_min) & (fft_frequencies <= freq_max))[0]
    if len(indices) == 0:
        # If no frequencies fall in that band, just return data unchanged
        return rotated_array
    start_idx, end_idx = indices[0], indices[-1]

    # Decide on the 'mode' for partial shifts. 'wrap' = periodic; 'nearest' or 'constant' also possible.
    shift_mode = 'wrap' if wrap else 'nearest'

    for i in range(num_rows):
        # Row-based shift formula. Negative sign reverses direction if desired.
        shift_amount = -(i - (num_rows // 2)) * shift_scale

        # Extract just the band to be shifted
        segment = data_array[i, start_idx:end_idx+1]

        # Shift that band by shift_amount. Because shift_amount might be float,
        shifted_segment = ndimage.shift(segment, shift=shift_amount, mode=shift_mode)

        # Place the shifted segment back into the array
        rotated_array[i, start_idx:end_idx+1] = shifted_segment

    return rotated_array

from pathlib import Path

# ── absolute root to the high-resolution FFT data ─────────────────────────
MODEL_ROOT = Path(
    "/jumbo/fitzlab/code/nl_nh_dimer/fresh_start/"
    "model/nl_nh_dimer_def/zorro_physics_more_res/data"
).resolve()

def get_th_zorro_plot(ax, epsilon_dBm):

    omega_c = 6.027e9  # Central frequency in Hz

    # Load data
    # directory = f'../data/theory_better_res/zorro_plots/epsilon_{epsilon_dBm}_dBm/freq/'
    # smb://jumbo.thayer.dartmouth.edu/jumbo/fitzlab/code/nl_nh_dimer/fresh_start/model/nl_nh_dimer_def/zorro_physics_more_res/data/epsilon_0_dBm
    # directory = f'../../../model/nl_nh_dimer_def/zorro_physics_more_res/data/epsilon_{epsilon_dBm}_dBm/freq'
    directory = MODEL_ROOT / f"epsilon_{epsilon_dBm}_dBm" / "freq"

    fft_frequencies, drive_frequencies, data_array = load_and_prepare_data(directory)

    # Apply targeted rotation
    data_array = rotate_data_within_range(fft_frequencies, data_array, omega_c / 1e9)

    extent = [fft_frequencies.max(), fft_frequencies.min(), drive_frequencies.min()/1e9, drive_frequencies.max()/1e9]
    # Assuming data_array and extent are defined earlier in your code
    im = ax.imshow(data_array, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=-60)

    ax.set_xlabel(r'Meas. Freq. [GHz]', fontsize=30)

    ax.set_xlim((omega_c-8e6)/1e9, (omega_c+8e6)/1e9)
    ax.set_ylim((omega_c-8e6)/1e9, (omega_c+8e6)/1e9)

    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=15)

    if epsilon_dBm == 0:
        ax.plot(fft_frequencies, fft_frequencies, color='darkblue', ls='--', lw=3.0, markeredgecolor='white',  markeredgewidth=1.5)
    else:
        ax.plot(fft_frequencies, fft_frequencies, color='crimson', ls='--', lw=3.0, markeredgecolor='white',  markeredgewidth=1.5)

    desired_ticks_plot = [6.022, 6.028, 6.034]
    closest_freqs_fft = desired_ticks(fft_frequencies, desired_ticks_plot)
    closest_freqs_drive = desired_ticks(drive_frequencies/1e9, desired_ticks_plot)

    ax.set_xticks(closest_freqs_fft)
    ax.set_xticklabels([f'${tick:.3f}$' for tick in desired_ticks_plot])
    ax.set_yticks(closest_freqs_drive)
    ax.set_yticklabels([f'${tick:.3f}$' for tick in desired_ticks_plot])

    ## both ticks should share the same freq. axis from exp.
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=30)

    return ax
