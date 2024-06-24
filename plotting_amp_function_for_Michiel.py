import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib import gridspec
import matplotlib.font_manager as fm

plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

plot_individual = False

# Define the function f(Pi) for given parameters
def f(Pi, a, b, c):
    return a * (1 / (1 + (Pi / b))) + c

# Define the piecewise function eta(Pi) for given parameters
def eta(Pi, a, b, c, eta_0, P_sat):
    return np.where(Pi <= P_sat, eta_0, f(Pi, a, b, c))

# Parameters for the two cases
params_G = {'a': 9.2448, 'b': 0.0086, 'c': 0.0, 'eta_0': 8.2716, 'P_sat': 0.0009981}
params_R = {'a': -0.3554, 'b': 0.00634, 'c': 0.7363, 'eta_0': 0.4854, 'P_sat': 0.0027981}

# Generate values for Pi
Pi_values = np.linspace(0, 25e-3, 400)

# Compute eta for each Pi for both cases
eta_values_G = eta(Pi_values, **params_G)
eta_values_R = eta(Pi_values, **params_R)

if plot_individual:
    # Plotting separate figures
    plt.figure(figsize=(8, 4))
    plt.plot(Pi_values, eta_values_G, label=r'$\eta(\Delta G)$', lw=2.0)
    plt.axvline(params_G['P_sat'], color='red', linestyle='--', label='$P_{sat}$')
    plt.xlabel('$P_i$ [Watts]')
    plt.ylabel('$\eta(P_i)$')
    plt.title('Case 1: Gain Function $\eta(\Delta G)$')
    plt.legend()
    plt.savefig('Delta_G.png')

    plt.figure(figsize=(8, 4))
    plt.plot(Pi_values, eta_values_R, label=r'$\eta(R_G)$', lw=2.0)
    plt.axvline(params_R['P_sat'], color='red', linestyle='--', label='$P_{sat}$')
    plt.xlabel('$P_i$ [Watts]')
    plt.ylabel('$\eta(P_i)$')
    plt.title('Case 2: Reflections off the Amplifier $\eta(R_G)$')
    plt.legend()
    plt.savefig('R_G.png')
else:
    pass

# Plotting combined figures with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(Pi_values, eta_values_G, lw=2.5)
ax1.axvline(params_G['P_sat'], color='red', linestyle='--', label=r'$P_{\rm{sat}, G}$')
ax1.set_ylabel(r'$\eta(\Delta G)$')
ax1.legend()

ax2.plot(Pi_values, eta_values_R, lw=2.5)
ax2.axvline(params_R['P_sat'], color='red', linestyle='--', label=r'$P_{\rm{sat}, R}$')
ax2.set_xlabel('$P_i$ [Watts]')
ax2.set_ylabel(r'$\eta(R_G)$')
ax2.legend()

plt.tight_layout()
plt.savefig('combined_transmission_reflection_amp.png')
plt.close()
