import numpy as np
from scipy.optimize import root, fsolve, root_scalar, least_squares
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import approx_fprime
from scipy.integrate import solve_ivp
import json

latex_rendering = False

if latex_rendering:
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Light']
    plt.rcParams['text.usetex'] = True
else:
    pass

with open('params.json', 'r') as f:
    parameters = json.load(f)

omega1 = parameters["omega1"] * 1e9
omega2 = parameters["omega2"] * 1e9
kappa_drive = parameters["kappa_drive"] * 1e6
kappa_readout = parameters["kappa_readout"] * 1e6
kappa_int_1 = parameters["kappa_int_1"] * 1e6
kappa_int_2 = parameters["kappa_int_2"] * 1e6
kappa_c = parameters["kappa_c"] * 1e6
beta = parameters["beta"] * 1e6
reflections_amp = parameters["reflections_amp"]
J0 = parameters["J0"] * 1e6

### Total baseline dissipation rates.
kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

h_bar = 1.054571817e-34
epsilon_dBm = -30

def J(net_gain):
    return 10**((net_gain) / 20) * kappa_c

def kappa_T(J_val, kappa_0):
    return 2*kappa_0 - J_val

def f(phi):
    return 1j*J0*(np.cos(phi/2)**2)*np.exp(1j*phi/2)

def model_function(x, b, x_sat):
    return (b + h_bar * omega1 * x_sat * kappa_c)/(b + h_bar * omega1 * x * kappa_c)

def piece_wise_amp_function(x, b, flat_line_value, x0):
    return np.where(x <= x0, flat_line_value, model_function(x, b, x0))

### Gain of the amplifier in linear operation
G_0 = 20.146 # dB
eta_0 = 10**(G_0/20) # converted to linear scale
params_G = {'eta_0': eta_0, 'b': 8.6e-3, 'P_sat': 0.9981e-3}

P_sat = params_G['P_sat']
b_amp = params_G['b']
eta_0 = params_G['eta_0']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c) # alpha_sat

b_1, flat_line_value_1, x0_1 = b_amp, eta_0, alpha_sat
b_2, flat_line_value_2, x0_2 = b_amp, eta_0, alpha_sat

def func(t, alpha, omega_d, phase, net_gain, epsilon_dBm):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
    epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))

    N1 = np.sqrt(alpha1_c.real**2 + alpha1_c.imag**2)**2
    N2 = np.sqrt(alpha2_c.real**2 + alpha2_c.imag**2)**2

    max_net_gain = 8.4  # dB
    attenuation = (max_net_gain - net_gain)
    IL = 11.746  # dB. Computed as the insertion loss at max delta G. IL = G_0 - max_net_gain = 20.146 - 8.4 = 11.746 dB
    eta_A = 10**(-(attenuation + IL)/20)

    J12 = piece_wise_amp_function(N1, b_1, flat_line_value_1, x0_1) * eta_A * kappa_c
    J21 = piece_wise_amp_function(N2, b_2, flat_line_value_2, x0_2) * eta_A * kappa_c

    kappa_diag_1 = kappa_T(J12, kappa_T_1)
    kappa_diag_2 = kappa_T(J21, kappa_T_2)

    d_alpha1 = -(1j*(omega1 - omega_d) + kappa_diag_1)*alpha1_c - (1j * J12 + f(phase)) * np.exp(-1j* phase)*alpha2_c + epsilon
    d_alpha2 = -(1j*(omega2 - omega_d) + kappa_diag_2)*alpha2_c - (1j * J21 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

min_freq = 5.975
max_freq = 6.085

## Conver to GHz
frequencies = np.linspace(min_freq*1e9, max_freq*1e9, 10)

transmission_dB = []
phase = np.pi #Just select a phase value
net_gain = 8.4 #Select a Net Gain value

plot_folder = f'time_domain_phase_{phase:.2f}_net_gain_{net_gain}'
os.makedirs(plot_folder, exist_ok=True)

for omega_d in frequencies:

    # Initial conditions. They are fixed for now, but we can try different (or even random) ones
    y0 = [1.0, 0.0, 1.0, 0.0]

    # Time span is set so that we time evolve for 30 times the characteristic dissipation rate kappa_c
    t_span = (0, 30*(1/kappa_c))

    # Solve the ODE
    sol = solve_ivp(
        func, 
        t_span, 
        y0, 
        args=(omega_d, phase, net_gain, epsilon_dBm), 
        method='RK45',
        dense_output=True,
        atol=1e-6,
        rtol=1e-3
    )

    # Plotting
    t = np.linspace(t_span[0], t_span[1], 1000)
    y = sol.sol(t) # this returns alpha1_real, alpha1_imag, alpha2_real, alpha2_imag

    ## Real and imaginary components of the solution (for alpha_2)
    alpha2_real = y[2] ## real
    alpha2_imag = y[3] ## imag

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Time evolution plot
    axs[0].plot(t*1e6, alpha2_real, label=r'Re[$\alpha_2$]', lw=2.0)
    axs[0].plot(t*1e6, alpha2_imag, label=r'Im[$\alpha_2$]', lw=2.0)
    axs[0].set_xlabel(r'$t \ [\mu s]$')
    axs[0].set_ylabel(r'$\alpha_2$')
    axs[0].legend(loc='best', fontsize=20)

    # Phase space plot
    axs[1].plot(alpha2_real, alpha2_imag, label=r'Phase space of $\alpha_2$', color='crimson', lw=2.0)
    axs[1].set_xlabel(r'Re[$\alpha_2$]')
    axs[1].set_ylabel(r'Im[$\alpha_2$]')

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/combined_plot_omega_d_{omega_d/1e9:.3f}_GHz.png')
    plt.close()
