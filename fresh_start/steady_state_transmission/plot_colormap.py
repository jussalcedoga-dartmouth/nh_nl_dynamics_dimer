import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

# Directory containing the CSV files
data_dir = 'single_traces_Vincent_hermitian/data'

# Collect all CSV file paths and their corresponding gain values
files = []
gains = []
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        gain = float(filename.split('_')[-1][:-4])  # Extract gain from filename
        files.append(os.path.join(data_dir, filename))
        gains.append(gain)

# Sort the files by gain values, descending for plot orientation
files = [x for _, x in sorted(zip(gains, files), key=lambda pair: pair[0], reverse=True)]
gains.sort(reverse=True)

# Prepare to collect all data
frequency = None
power_db_matrix = []

# Read each file and extract the necessary data
for file in files:
    df = pd.read_csv(file)
    if frequency is None:
        frequency = df['Frequency'].values/1e9
    power_db_matrix.append(df['Power_dB'].values)

# Convert the list to a NumPy array for plotting
power_db_matrix = np.array(power_db_matrix)

# Plotting
fig, ax = plt.subplots(figsize=(7, 7))
c = ax.imshow(power_db_matrix, aspect='auto', cmap='inferno', 
            #   extent=[frequency[0], frequency[-1], gains[-1], gains[0]], vmin=-50)
            extent=[frequency[0], frequency[-1], gains[-1], gains[0]])

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel(r'$\Delta G$')
fig.colorbar(c, ax=ax, label=r'$S_{21}$ [dB]')
plt.tight_layout()
plt.savefig('transmission_hermitian.png')
plt.close()

## Plot the max as a function of Delta G...

# Sort the files by gain values, descending for plot orientation
files = [x for _, x in sorted(zip(gains, files), key=lambda pair: pair[0], reverse=True)]
gains.sort(reverse=True)

# Prepare to collect all data
max_power_dB = []

# Read each file and extract the necessary data
for file in files:
    df = pd.read_csv(file)
    max_power_dB.append(df['Power_dB'].max())  # Find the maximum power dB for each gain

# Plotting the maximum transmission as a function of Net Gain
plt.figure(figsize=(7, 4))
plt.plot(gains, max_power_dB, marker='o', linestyle='-', color='crimson')
plt.xlabel(r'$\Delta G$')
plt.ylabel(r'$S_{21}^{\rm{max}}$ [dB]')
plt.tight_layout()
plt.savefig('max_transmission_vs_gain_hermitian.png')
plt.close()
