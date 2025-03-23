import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

# plt.rcParams.update({'font.size': 25})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = True

def set_yaxis_ticks(ax, tick_size=45):
    # Set major locator to MultipleLocator (pi)
    ax.set_ylim(0, 2*np.pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
    # Define custom formatter to handle specific label formatting
    def format_func(value, pos):
        if np.isclose(value, 0):
            return '0'
        elif np.isclose(value, np.pi):
            return r'$\pi$'  # Display 'π' instead of '1π'
        elif np.isclose(value, 2 * np.pi):
            return r'2$\pi$'  # Optionally handle the 2π case as well
        else:
            return f'{value/np.pi:.1g}$\pi$'  # General case

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # Increase tick size
    ax.tick_params(axis='y', which='major', labelsize=tick_size)

# List of epsilons and their corresponding fixed colors for clarity using a colormap
epsilons = ['None', '0', '4', '8', '12', '16']
colors = cm.inferno_r(np.linspace(0, 1, len(epsilons)))

# Prepare the figure
fig, ax = plt.subplots(figsize=(8, 8))

# Loop through each epsilon, read the data, and plot contours where the Flag is 1.0
for idx, epsilon in enumerate(epsilons):
    csv_filename = f'epsilon_contours_data/results_epsilon_{epsilon}_dBm.csv'
    
    # Load data from CSV
    df = pd.read_csv(csv_filename)
    
    # Assuming the CSV has these columns: 'Gain [dB]', 'Phase [rads.]', 'Flag'
    gain_values = np.sort(df['Gain [dB]'].unique())
    phase_values = np.sort(df['Phase [rads.]'].unique())
    
    # Create a matrix to hold the flag data, NaN initially
    imshow_data = np.full((len(phase_values), len(gain_values)), np.nan)
    
    # Fill in the matrix where the flag is 1.0
    for _, row in df.iterrows():
        if row['Flag'] == 1.0:
            phase_index = np.where(phase_values == row['Phase [rads.]'])[0][0]
            gain_index = np.where(gain_values == row['Gain [dB]'])[0][0]
            imshow_data[phase_index, gain_index] = 1  # We only care about Flag = 1.0

    # Plot the data
    ax.contourf(gain_values, phase_values, imshow_data, levels=[0.5, 1.5], colors=[colors[idx]], alpha=0.6)

# Create custom legend handles
legend_handles = [Patch(facecolor=colors[i], edgecolor='none', label=f'{eps} dBm' if eps != 'None' else 'None') for i, eps in enumerate(epsilons)]

ax.legend(handles=legend_handles, title="Drive Power", loc='upper left', fontsize=12, title_fontsize=12)
ax.set_xlabel(r'$\Delta G$ [dB]', fontsize=45)
ax.set_ylabel(r'$\phi$', fontsize=50)
ax.tick_params(axis='both', which='major', labelsize=40)
set_yaxis_ticks(ax)
ax.set_xlim(4.01, ax.get_xlim()[1])
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.tight_layout()
plt.savefig('epsilon_contours.png', dpi=400)
plt.close()
