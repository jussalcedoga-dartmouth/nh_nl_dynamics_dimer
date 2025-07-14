import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# Load parameters
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

omega_c = omega1
h_bar = 1.054571817e-34

params_G = {'b': 8.6e-3, 'P_sat': 0.9981e-3}
P_sat = params_G['P_sat']
b_amp = params_G['b']

alpha_sat = P_sat / (h_bar * omega1 * kappa_c)

kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_readout + kappa_c

def kappa_T(J_val, kappa_0):
    return 2 * kappa_0 - J_val

def f(phi):
    return 1j * J0 * np.cos(phi / 2) * np.exp(1j * phi / 2)

def J_nl(alpha, net_gain):
    prefactor = kappa_c * 10 ** (net_gain / 20)
    if alpha <= alpha_sat:
        return prefactor * 1.0
    else:
        numerator = b_amp + h_bar * omega2 * alpha_sat * kappa_c
        denominator = b_amp + h_bar * omega2 * alpha * kappa_c
        return prefactor * (numerator / denominator)

def func(t, alpha, phase, net_gain, omega_d, epsilon_dBm):
    alpha1, alpha1_i, alpha2, alpha2_i = alpha

    epsilon = 0.0
    if epsilon_dBm is not None:
        epsilon_watts = 10 ** ((epsilon_dBm - 30) / 10)
        epsilon = np.sqrt((kappa_drive * epsilon_watts) / (h_bar * omega_d))

    alpha1_c = alpha1 + 1j * alpha1_i
    alpha2_c = alpha2 + 1j * alpha2_i

    N1 = abs(alpha1_c) ** 2
    N2 = abs(alpha2_c) ** 2

    J_nl_1 = J_nl(N1, net_gain)
    J_nl_2 = J_nl(N2, net_gain)

    kappa_diag_1 = kappa_T(J_nl_1, kappa_T_1)
    kappa_diag_2 = kappa_T(J_nl_2, kappa_T_2)

    d_alpha1 = -(kappa_diag_1) * alpha1_c - (1j * J_nl_1 + f(phase)) * np.exp(-1j * phase) * alpha2_c + epsilon
    d_alpha2 = -(kappa_diag_2) * alpha2_c - (1j * J_nl_2 + f(phase)) * alpha1_c

    return [d_alpha1.real, d_alpha1.imag, d_alpha2.real, d_alpha2.imag]

def simulate_case(args):
    net_gain, phase, epsilon_dBm = args

    y0 = [1.0e7, 0.0, 1.0e7, 0.0]
    t_span = (0, 10000 * (1 / kappa_c))
    omega_d = omega1

    sol = solve_ivp(
        func, t_span, y0,
        args=(phase, net_gain, omega_d, epsilon_dBm),
        method='RK45', dense_output=True,
        atol=1e-6, rtol=1e-3
    )

    t = np.linspace(t_span[0], t_span[1], 10000)
    y = sol.sol(t)
    signal = y[2] + 1j * y[3]

    start_index = int(0.2 * len(t))
    t = t[start_index:]
    signal = signal[start_index:]

    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))
    fft_norm = fft_result / len(signal)
    psd = fft_norm.real ** 2 + fft_norm.imag ** 2

    fft_freq_shifted = np.fft.fftshift(fft_freq) * 2 * np.pi
    psd_shifted = np.fft.fftshift(psd)

    power_watts = psd_shifted * kappa_readout * h_bar * omega2
    power_dBm = 10 * np.log10(power_watts * 1e3)
    freq_axis = -fft_freq_shifted / (2 * np.pi * 1e6) + omega_d / 1e6

    return (epsilon_dBm, phase, freq_axis, power_dBm)

# Define the main execution
if __name__ == '__main__':
    net_gain = 8.4
    drive_strengths = [None, 8, 12, 18]
    phases = [np.pi - 0.5, np.pi + 0.5]
    tasks = [(net_gain, phase, dBm) for phase in phases for dBm in drive_strengths]

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(simulate_case, tasks), total=len(tasks)))

    # ─────────────────────────  PLOTTING  ───────────────────────────
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    # --- match experimental styling ------------------------------------------------
    plt.rcParams.update({
        "font.size": 30,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Light"],
        "text.usetex": True,
    })

    COLORS = {None: "black", 8: "royalblue", 12: "crimson", 18: "darkorange"}
    PHASE_LABELS = {
        0: r"$\phi = \pi - 0.5\\ \Delta G = 8.4\,\mathrm{dB}$",
        1: r"$\phi = \pi + 0.5\\ \Delta G = 8.4\,\mathrm{dB}$",
    }

    loc = MaxNLocator(nbins=5)
    fmt = FormatStrFormatter(r"$%.3f$")

    fig, axs = plt.subplots(
        nrows=len(drive_strengths), ncols=len(phases),
        figsize=(20, 16),
        sharex=False, sharey=False,
        gridspec_kw={"wspace": 0.40, "hspace": 0.20}
    )

    for ε_dBm, phase, freq_axis, power_dBm in results:
        col = 0 if phase < np.pi else 1
        row = drive_strengths.index(ε_dBm)
        ax  = axs[row, col]

        # --- trace ---------------------------------------------------
        ax.plot(freq_axis / 1e3, power_dBm, lw=3.0, color=COLORS[ε_dBm])

        # x-axis limits identical to experiment
        # ax.set_xlim(5.99, 6.052)

        # drive-power label inside panel
        label = "None" if ε_dBm is None else f"{ε_dBm} dBm"
        ax.annotate(
            label, xy=(0.02, 0.80), xycoords="axes fraction",
            fontsize=28, color=COLORS[ε_dBm],
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6)
        )

        # y-label every row
        ax.set_ylabel("Power [dBm]")

        # tick formatting
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        if row < len(drive_strengths) - 1:
            ax.set_xticklabels([])

        # vertical dashed line at resonance
        ax.axvline(omega_c / 1e9, ls="--", lw=3.0, color="k", alpha=0.5)

    # --- phase (column) annotations --------------------------------------------
    axs[0, 0].annotate(
        PHASE_LABELS[0],
        xy=(0.02, 0.45), xycoords="axes fraction",
        fontsize=28, color="black",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6)
    )
    axs[0, 1].annotate(
        PHASE_LABELS[1],
        xy=(0.02, 0.45), xycoords="axes fraction",
        fontsize=28, color="black",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6)
    )

    # --- global panel labels ----------------------------------------------------
    axs[0, 0].annotate(r"\textbf{{a}}", xy=(-0.15, 1.1), xycoords="axes fraction",
                    fontsize=50, fontweight="bold")
    axs[0, 1].annotate(r"\textbf{{b}}", xy=(-0.15, 1.1), xycoords="axes fraction",
                    fontsize=50, fontweight="bold")

    # bottom-row x-labels
    for c in range(2):
        axs[-1, c].set_xlabel("Frequency [GHz]")

    # left, bottom, right, top
    fig.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.98)
    fig.savefig("dimer_emission_Pd_theo_panels.png", dpi=300)
    print("✓ Figure saved → dimer_spectra_grid.png")