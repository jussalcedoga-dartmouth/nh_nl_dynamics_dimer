#!/usr/bin/env python3
"""
plot_dimer_emission_2panels.py  — experimental spectra (phase‑query fixed)
────────────────────────────────────────────────────────────────────────────
Minimal change to the **user‑supplied single‑phase script**: we now query
**exactly** the two absolute phases derived the same way the user did in
manual runs — no offset math guesswork.

Phase selection recipe (identical to user’s snippet):
    φ_ref = 63°
    φ_desired = rad2deg(π ± 0.5)
    φ_abs = (φ_ref + φ_desired)°  →  radians

Everything else — fonts, linewidths, axis limits, annotations — remains
verbatim.  The figure is a 4 × 2 grid (16 × 14 in).
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ── CONSTANTS & USER SETTINGS ─────────────────────────────────────────────
SMB_ROOT = Path(
    "/jumbo/fitzlab/code/QM_fitzlab/240729_Phase_Diagram_Drive/"
    "fig4_combs_contours_overnight"
)
GAIN_TARGET = 8.4
PD_LIST     = [None, 4, 8, 12]
PHI_REF_DEG = 63
PHASE_TOL   = 0.03  # radians tolerance
COLORS      = ["black", "royalblue", "crimson", "darkorange"]

plt.rcParams.update({
    "font.size": 30,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ── Two absolute target phases (user’s formula) ───────────────────────────
PHI_DESIRED_DEG = [np.rad2deg(np.pi - 0.5) % 360,  # π − 0.5
                   np.rad2deg(np.pi + 0.5) % 360]  # π + 0.5
PHASES_RADS = [np.deg2rad((PHI_REF_DEG + d) % 360) for d in PHI_DESIRED_DEG]
PHASE_LABELS = {
    0: r"$\phi = \pi - 0.5\\ \Delta G = 8.4 \mathrm{{dB}}$",
    1: r"$\phi = \pi + 0.5\\ \Delta G = 8.4 \mathrm{dB}$",
}

# ── REGEX helpers ─────────────────────────────────────────────────────────
re_eps   = re.compile(r"epsilon_(\d+|None)")
re_phase = re.compile(r"phase_([\d.]+)")

# ── Gather CSVs: csv_paths[φ_abs][Pd] = path ──────────────────────────────
csv_paths = {phi: {} for phi in PHASES_RADS}
for root, _, files in os.walk(SMB_ROOT):
    m_eps, m_phase = re_eps.search(root), re_phase.search(root)
    if not m_eps or not m_phase:
        continue

    pd_val = None if m_eps.group(1) == "None" else int(m_eps.group(1))
    if pd_val not in PD_LIST:
        continue

    phase_dir_rad = float(m_phase.group(1))

    for phi_target in PHASES_RADS:
        if abs(phase_dir_rad - phi_target) > PHASE_TOL:
            continue
        for f in files:
            if f.startswith("combs_characterization") and f.endswith(".csv") and f"gain_{GAIN_TARGET}" in f:
                csv_paths[phi_target][pd_val] = Path(root)/f

# sanity check
for phi, sub in csv_paths.items():
    if len(sub) < len(PD_LIST):
        missing = [p for p in PD_LIST if p not in sub]
        raise RuntimeError(f"Missing Pd {missing} for phase {phi:.3f} rad")

# ── Plotting (unchanged style) ────────────────────────────────────────────
loc = MaxNLocator(nbins=5)
fmt = FormatStrFormatter(r"$%.3f$")

fig, axs = plt.subplots(
    nrows=len(PD_LIST), ncols=2,
    figsize=(20, 16),
    sharex=False, sharey=False,
    gridspec_kw={"wspace": 0.40, "hspace": 0.20},
)

for col, phi in enumerate(PHASES_RADS):
    for row, pd_val in enumerate(PD_LIST):
        ax = axs[row, col]
        df = pd.read_csv(csv_paths[phi][pd_val])
        freq_GHz = df["frequency"].values/1e9
        amp      = df["trace_corrected"].values

        ax.plot(freq_GHz, amp, color=COLORS[row], lw=3.0)
        ax.set_xlim(5.99, 6.052)

        label = "None" if pd_val is None else f"{pd_val} dBm"
        ax.annotate(label, xy=(0.02,0.80), xycoords="axes fraction",
                    fontsize=28, color=COLORS[row], ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

        ax.set_ylabel("Power [dBm]")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        
        if row < len(PD_LIST)-1:
            ax.set_xticklabels([])

        bare_freq = (6.034e9+6.036e9)/2 - 7.5e6
        ax.axvline(bare_freq/1e9, ls="--", lw=3.0, color="k", alpha=0.5)

    axs[0, col].annotate(PHASE_LABELS[col], xy=(0.02,0.40), xycoords="axes fraction",
                          fontsize=28, ha="left", va="center",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

# --- global panel labels ----------------------------------------------------
axs[0, 0].annotate(r"\textbf{{a}}", xy=(-0.15, 1.1), xycoords="axes fraction",
                fontsize=50, fontweight="bold")
axs[0, 1].annotate(r"\textbf{{b}}", xy=(-0.15, 1.1), xycoords="axes fraction",
                fontsize=50, fontweight="bold")

for c in range(2):
    axs[-1,c].set_xlabel("Frequency [GHz]")

# plt.tight_layout(rect=[0,0,0.97,0.94])
fig.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.98)

out = Path("../plots/dimer_emission_Pd_exp_panels.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=300)
print("✓ Figure saved →", out.resolve())
