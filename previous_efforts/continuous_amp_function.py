import numpy as np
from matplotlib import pyplot as plt

# Update matplotlib settings for better visual aesthetics
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Light'],
    'text.usetex': True
})

# Define function f(Pi) for given parameters
def f(Pi, a, b):
    return a * (1 / (1 + (Pi / b)))

# Define the piecewise function eta(Pi) for continuity
def eta(Pi, eta_0, b, P_sat):
    a = eta_0 * (1 + (P_sat / b))  # Ensure 'a' guarantees continuity at P_sat
    print(a)
    return np.where(Pi <= P_sat, eta_0, f(Pi, a, b))

# Define parameters
G_0 = 10**(20.146/20)
params_G = {'eta_0': G_0, 'b': 8.6e-3, 'P_sat': 0.9981e-3}

# Generate Pi values and compute eta for each Pi
Pi_values = np.linspace(0, 25e-3, 400)
eta_values_G = eta(Pi_values, **params_G)

# Create plot
plt.figure(figsize=(8, 4))
plt.plot(Pi_values, eta_values_G, label=r'$\eta(\Delta G)$', lw=2.0)
plt.axvline(params_G['P_sat'], color='red', linestyle='--', label='$P_{\rm{sat}}$')
plt.xlabel('$P_i$ [Watts]')
plt.ylabel('$\eta(P_i)$')
plt.title('Gain Function $\eta(\Delta G)$')
plt.legend()
plt.tight_layout()
plt.savefig('Delta_G_continuous.png')
plt.close()
