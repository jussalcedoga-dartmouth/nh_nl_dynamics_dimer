#!/usr/bin/env python3
"""
plot_dimer_emission_subfigs.py
──────────────────────────────
Search the SMB tree for ΔG = 8.4 dB, φ ≈ 2π/3 (+63° reference),
Pd = None, 4, 8, 12 dBm and plot emission spectra (trace_corrected vs frequency).
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ── CONSTANTS & USER SETTINGS ──────────────────────────────────────────────
SMB_ROOT = Path(
    "/jumbo/fitzlab/code/QM_fitzlab/240729_Phase_Diagram_Drive/"
    "fig4_combs_contours_overnight"
)

GAIN_TARGET = 8.4
PD_LIST     = [None, 4, 8, 12]  # Include None to represent 'no drive'
PHI_REF_DEG = 63
PHI_DESIRED = np.rad2deg(np.pi + 0.5) % 360  # You can change to 2π/3 if needed
PHI_ABS_RAD = (PHI_REF_DEG + PHI_DESIRED) * np.pi / 180
PHASE_TOL   = 0.03
COLORS      = ["black", "royalblue", "crimson", "darkorange"]  # Color for each Pd

plt.rcParams.update({
    "font.size": 32,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ── REGEX HELPERS ──────────────────────────────────────────────────────────
re_eps   = re.compile(r"epsilon_(\d+|None)")
re_phase = re.compile(r"phase_([\d.]+)")
re_gain  = re.compile(r"gain_(\d+(?:\.\d+)?)")

# ── FIND CSVs ──────────────────────────────────────────────────────────────
csv_paths = {}  # Pd → Path
for root, _, files in os.walk(SMB_ROOT):
    # epsilon filter
    m_eps = re_eps.search(root)
    if not m_eps:
        continue
    eps_str = m_eps.group(1)
    pd_val = None if eps_str == "None" else int(eps_str)
    if pd_val not in PD_LIST:
        continue

    # phase filter
    m_phase = re_phase.search(root)
    if not m_phase:
        continue
    phase_val = float(m_phase.group(1))
    if abs(phase_val - PHI_ABS_RAD) > PHASE_TOL:
        continue

    # find matching CSV
    for f in files:
        if not f.startswith("combs_characterization") or not f.endswith(".csv"):
            continue
        if f"gain_{GAIN_TARGET}" not in f:
            continue
        csv_paths[pd_val] = Path(root) / f

if not csv_paths:
    raise RuntimeError("No matching CSVs found.")

print("Matched files:")
for k in sorted(csv_paths, key=lambda x: (x is not None, x)):
    label = "no drive" if k is None else f"{k} dBm"
    print(f"  Pd = {label}  →  {csv_paths[k]}")

# ── LOAD & PLOT ────────────────────────────────────────────────────────────
loc = MaxNLocator(nbins=5)
fmt = FormatStrFormatter(r"$%.3f$")

fig = plt.figure(figsize=(12, 13))
gs  = gridspec.GridSpec(len(PD_LIST), 1, height_ratios=[1]*len(PD_LIST), hspace=0.15)

for i, pd_val in enumerate(PD_LIST):
    if pd_val not in csv_paths:
        continue

    df = pd.read_csv(csv_paths[pd_val])
    freq_GHz = df["frequency"].to_numpy() / 1e9
    amp      = df["trace_corrected"].to_numpy()

    ax = fig.add_subplot(gs[i, 0])
    ax.plot(freq_GHz, amp, color=COLORS[i], lw=3.0)
    ax.set_xlim(5.99, 6.052)

    label = "None" if pd_val is None else f"{pd_val} dBm"
    ax.annotate(label,
                xy=(0.02, 0.8), xycoords="axes fraction",
                fontsize=30, color=COLORS[i],
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

    ax.set_ylabel("Power [dBm]")

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)

    if i < len(PD_LIST) - 1:
        ax.set_xticklabels([])

    loading = 7.5e6
    bare_cavity_freq = np.mean([6.034e9, 6.036e9]) - loading
    ax.axvline(bare_cavity_freq / 1e9, ls='--', lw=3.0, color='k', alpha=0.5)

# bottom-axis label
fig.axes[-1].set_xlabel("Frequency [GHz]")

# Add title annotation
fig.axes[0].annotate(r"$\phi = \pi + 0.5\\ \Delta G = 8.4\,\mathrm{dB}$",
            xy=(0.02, 0.4), xycoords="axes fraction",
            fontsize=30, color='black',
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

plt.tight_layout()
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(right=0.95)

out = Path("../plots/dimer_emission_Pd_exp_plus.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=300)
print(f"✓ Figure saved → {out.resolve()}")
