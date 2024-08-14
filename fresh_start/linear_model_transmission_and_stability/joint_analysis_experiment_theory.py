import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from load_exp_data import *
from get_model_data import *

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

# Define the output folder
output_folder = 'output_photonic_dimer_J_0_notes_model'
os.makedirs(output_folder, exist_ok=True)

# Define parameters and load data
min_freq = 5.975
max_freq = 6.085

frequencies = np.linspace(min_freq*1e9, max_freq*1e9, hermitian_num_points)
frequencies_nh = np.linspace(min_freq*1e9, max_freq*1e9, hermitian_num_points)

phase_hermitian = 'hermitian'
phase_nonhermitian = 'nonhermitian'

## hermitian transmission
transmissions_sorted, net_gains_sorted = transmission_vs_frequency(phase_hermitian, min_freq, max_freq)
alpha2_solutions, net_gains = get_data_for_phase(frequencies, net_gains_sorted, 0, epsilon_dBm=-30)

net_gains, differences, max_values_below, max_values_above = get_metrics_spectrum(f'../experiment_analysis/data/experiment/{phase_hermitian}')
net_gains_theory, differences_theory, max_values_below_theory, max_values_above_theory = get_metrics_spectrum_theory(alpha2_solutions, frequencies, net_gains, center_freq=6.028e9)

## non-hermitian transmission
transmissions_sorted_nh, net_gains_sorted_nh = transmission_vs_frequency(phase_nonhermitian, min_freq, max_freq)
alpha2_solutions_nh, net_gains_nh = get_data_for_phase(frequencies_nh, net_gains_sorted, np.pi, epsilon_dBm=-30)

net_gains_nh, max_values_nh = get_metrics_spectrum_nonhermitian(f'../experiment_analysis/data/experiment/{phase_nonhermitian}')
net_gains_theory, max_values_nh_theory = get_metrics_spectrum_nonhermitian_theory(alpha2_solutions_nh, frequencies_nh, net_gains_nh)

# Setup figure layout
fig = plt.figure(figsize=(22, 22))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.8])

# Colorbar settings based on the experimental data for each phase.
vmin, vmax = np.min(transmissions_sorted), np.max(transmissions_sorted)
vmin_nh, vmax_nh = np.min(transmissions_sorted_nh), np.max(transmissions_sorted_nh)

# Top row for Hermitian phase data
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(transmissions_sorted, aspect='auto', cmap='inferno', extent=[min_freq, max_freq, net_gains_sorted[0], net_gains_sorted[-1]], vmin=vmin, vmax=vmax)
plt.colorbar(im1, ax=ax1, label='$S_{21}$ [dB]')
ax1.set_title('Experimental Transmission')
ax1.set_xlabel('Frequency [GHz]')
ax1.set_ylabel('Net Gain [dB]')

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(alpha2_solutions, aspect='auto', cmap='inferno', origin='lower', extent=[min_freq, max_freq, net_gains[0], net_gains[-1]])
# im2 = ax2.imshow(alpha2_solutions, aspect='auto', cmap='inferno', origin='lower', extent=[min_freq, max_freq, net_gains[0], net_gains[-1]], vmin=vmin, vmax=vmax)
plt.colorbar(im2, ax=ax2, label='$S_{21}$ [dB]')
ax2.set_title('Theoretical Transmission')
ax2.set_xlabel('Frequency [GHz]')
ax2.set_ylabel('Net Gain [dB]')

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(net_gains, max_values_above, 'o', color='crimson', label='Exp')
ax3.plot(net_gains, max_values_below, 'o', color='royalblue')
ax3.plot(net_gains_theory, max_values_above_theory, '-', color='crimson', lw=2.0, label='Theory')
ax3.plot(net_gains_theory, max_values_below_theory, '-', color='royalblue', lw=2.0)
ax3.set_ylabel(r'$S_{21}^{{\rm{max}}}$ [dB]')
ax3.set_xlabel('Net Gain [dB]')
ax3.set_title('Max Transmission')
ax3.legend()

# Middle row for Non-Hermitian phase data
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(transmissions_sorted_nh, aspect='auto', cmap='inferno', extent=[min_freq, max_freq, net_gains_sorted_nh[0], net_gains_sorted_nh[-1]], vmin=vmin_nh, vmax=vmax_nh)
plt.colorbar(im4, ax=ax4, label='$S_{21}$ [dB]')
ax4.set_title('Experimental Transmission')
ax4.set_xlabel('Frequency [GHz]')
ax4.set_ylabel('Net Gain [dB]')

ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(alpha2_solutions_nh, aspect='auto', cmap='inferno', origin='lower', extent=[min_freq, max_freq, net_gains_nh[0], net_gains_nh[-1]], vmin=vmin_nh, vmax=vmax_nh)
# im5 = ax5.imshow(alpha2_solutions_nh, aspect='auto', cmap='inferno', origin='lower', extent=[min_freq, max_freq, net_gains_nh[0], net_gains_nh[-1]])
plt.colorbar(im5, ax=ax5, label='$S_{21}$ [dB]')
ax5.set_title('Theoretical Transmission')
ax5.set_xlabel('Frequency [GHz]')
ax5.set_ylabel('Net Gain [dB]')

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(net_gains_nh, max_values_nh, 'o', color='crimson', label='Exp')
ax6.plot(net_gains_nh, max_values_nh_theory, '-', color='crimson', lw=2.0, label='Theory')
ax6.set_xlabel('Net Gain [dB]')
ax6.set_ylabel(r'$S_{21}^{{\rm{max}}}$ [dB]')
ax6.set_title('Max Transmission')
ax6.legend()

# Bottom row exclusively for phase diagram
ax7 = fig.add_subplot(gs[2, 1:2])
get_phase_diagram(ax7)

parameters = get_used_parameters()
param_names = [r'$\omega_1$', r'$\omega_2$', r'$\kappa_{d}$', r'$\kappa_{r}$', r'$\kappa_{0,1}$', r'$\kappa_{0,2}$', r'$\kappa_c$', r'$\beta$', r'reflections_amp', r'$J_0$']
units = ['GHz', 'GHz', 'MHz', 'MHz', 'MHz', 'MHz', 'MHz', 'MHz', ' ', 'MHz']

formatted_params = ', '.join([f"{name} = {value} {unit}" for name, value, unit in zip(param_names, parameters, units)])
fig.suptitle(formatted_params, fontsize=20)

## for storing
param_tags = ['omega1', 'omega2', 'kappa_d', 'kappa_r', 'kappa_int1', 'kappa_int2', 'kappa_c', 'beta', 'reflections_amp', 'J_0']
formatted_tags = '_'.join([f"{tag}_{value:.2f}" for tag, value in zip(param_tags, parameters)])

## Let's store all of the different configurations with proper params formatting
filename = f"{output_folder}/complete_analysis_{formatted_tags}_J_0.png"
plt.tight_layout()
plt.savefig(filename)
plt.close()
