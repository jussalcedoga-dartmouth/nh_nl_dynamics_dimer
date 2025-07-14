#!/usr/bin/env python3
# coding: utf-8
"""
Plot two traces (“stable” @ 4 dB and “unstable” @ 8.4 dB) plus mark
the global maximum of the spectrum with a gold star and a vertical
dashed line, and display its value in the bottom-left corner.
"""

# ─────────────────────────── imports ──────────────────────────────────────
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ───────────────── absolute directories ───────────────────────────────────
ROOT = Path(
    "/jumbo/fitzlab/code/QM_fitzlab/240726_Phase_Diagram_Calibrated_Dimer"
).resolve()

SUBDIR_PATTERN = "so_phase_diagram_calibrated_dimer0/gain_{gain}_adaptive_offset"
PHASE_RE = re.compile(r"_phase_([0-9.]+)_rad_")

def find_file_by_phase(gain: float, target_phi: float, tol: float = 1e-3) -> Path:
    base = ROOT / SUBDIR_PATTERN.format(gain=f"{gain:g}")
    best, best_err = None, np.inf
    for f in base.glob("*_phase_*_rad_ext_att_20_no_driving.csv"):
        m = PHASE_RE.search(f.name)
        if m:
            err = abs(float(m.group(1)) - target_phi)
            if err < best_err:
                best, best_err = f, err
    if best is None or best_err > tol:
        raise FileNotFoundError(f"No CSV within ±{tol} rad of target phase in {base}")
    return best

# ───────────────── locate CSVs ────────────────────────────────────────────
stable_csv = ROOT / SUBDIR_PATTERN.format(gain="4.0") / \
    "240726190200_phase_gain_diagram_gain_4.0_dB_gain_4.0_dB_phase_4.328_rad_ext_att_20_no_driving.csv"

phi0   = np.deg2rad(65.0)                 # 65 °
phi_pi = (phi0 + np.pi) % (2*np.pi)
pi_csv = find_file_by_phase(8.4, phi_pi, tol=0.5)
print(f"π-shifted file found → {pi_csv.name}")

# ───────────────── load data ──────────────────────────────────────────────
stable_df   = pd.read_csv(stable_csv)
unstable_df = pd.read_csv(pi_csv)

f_stable   = stable_df["frequency"] / 1e9
p_stable   = stable_df["trace_corrected"]

f_unstable = unstable_df["frequency"] / 1e9
p_unstable = unstable_df["trace_corrected"]

# ───────────────── find global maximum (across both traces) ───────────────
all_freqs  = np.concatenate([f_stable,   f_unstable])
all_powers = np.concatenate([p_stable,   p_unstable])
idx_max    = np.argmax(all_powers)
freq_max   = all_freqs[idx_max]
amp_max    = all_powers[idx_max]

# ───────────────── plotting ───────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 20,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(f_stable,   p_stable,   color="royalblue", lw=2, label=r"$\Delta G = 4$ dB")
ax.plot(f_unstable, p_unstable, color="crimson",   lw=2, label=r"$\Delta G = 8.4$ dB")

# ── mark the maximum ──────────────────────────────────────────────────────
ax.plot(freq_max, amp_max, marker="*", color="k", markersize=8,
        label="_nolegend_")                       # gold star (no legend entry)
ax.axvline(freq_max, color="k", ls="--", lw=1.5, label="_nolegend_")

# annotation
ax.text(0.02, 0.15,
        f"Amplitude = {amp_max:.2f} dBm\nFrequency = {freq_max:.3f} GHz",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=16)

ax.text(0.02, 0.90, r"$\phi = \pi$", transform=ax.transAxes,
        ha="left", va="top", fontsize=20)

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Power [dBm]")
ax.set_xlim(5.985, 6.07)
ax.legend(loc="best", frameon=False, fontsize=15)
ax.xaxis.set_major_locator(MaxNLocator(5))
fig.tight_layout()

out = "../plots/referee_report_LC.png"
fig.savefig(out, dpi=300)
print(f"✓ Figure saved → {out}")
