#!/usr/bin/env python3
"""
Fig-2: compact layout, slim colour-bars, internal φ-labels.
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# --------------------------------------------------------------------------
# project-local helpers
# --------------------------------------------------------------------------
from get_plots_individually import *        # create_amplitude_plot, bottom panels
from return_phase_diagram import *          # noqa: F401

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ── PATHS & CONSTANTS ────────────────────────────────────────────────────
cwd             = os.getcwd()
data_base_path  = os.path.join(cwd, "..", "data")
experiment_path = os.path.join(data_base_path, "experiment")
theory_path     = os.path.join(data_base_path, "theory")

phases          = ["hermitian", "nonhermitian"]
data_types      = ["experiment", "theory"]
ext_atten       = 20
wanted_freqs    = [6.00, 6.03, 6.06]   # GHz  → y-ticks on colour-plots

# ── UTILITIES ─────────────────────────────────────────────────────────────
def load_data(fp, has_header):
    df = pd.read_csv(fp) if has_header else pd.read_csv(
        fp, header=None, names=["Frequency", "Power_dB"])
    if not has_header:
        df["Power_dB"] += ext_atten
    return df

def extract_net_gain(fname):
    return float(fname.split("_")[-1].replace(".csv", ""))

def nearest_idx(arr, val):
    return np.abs(arr - val).argmin()

# ── COLOUR-PLOT FACTORY ───────────────────────────────────────────────────
def colourplot(data, ax, phase_tag, data_type,
               gmin, gmax, label_bottom=False):
    gains = np.array([d[0] for d in data])
    trans = np.array([d[1]["Power_dB"].values for d in data])
    freqs = data[0][1]["Frequency"].values / 1e9      # → GHz

    mask  = (freqs >= 5.975) & (freqs <= 6.085)
    freqs = freqs[mask]
    trans = np.array([row[mask] for row in trans])

    vmin, vmax = (-39, -14) if phase_tag == "hermitian" else (-35, 28)

    m = ax.imshow(trans.T, aspect="auto",
                  extent=[gmin, gmax, freqs.min(), freqs.max()],
                  origin="lower", cmap="inferno",
                  vmin=vmin, vmax=vmax)

    # y-ticks
    yt = [freqs[nearest_idx(freqs, t)] for t in wanted_freqs]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"${t:.2f}$" for t in wanted_freqs])

    # annotations
    ax.axvline(4.78, ls="--", lw=3, color="white", alpha=0.5)
    phi = r"$\phi = 0$" if phase_tag == "hermitian" else r"$\phi = \pi$"
    ax.text(0.20, 0.20, phi, transform=ax.transAxes,
            ha="right", va="top", color="white", fontsize=25)
    tag = "experiment" if data_type == "experiment" else "numerics"
    ax.text(0.05, 0.90, tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="none", alpha=0.8))
    return m

# ── LOAD DATA ────────────────────────────────────────────────────────────
data_by_phase = {}
for ph in phases:
    data_by_phase[ph] = {}
    for dt in data_types:
        droot = experiment_path if dt == "experiment" else theory_path
        files = sorted(os.listdir(os.path.join(droot, ph)), key=extract_net_gain)
        data_by_phase[ph][dt] = [(extract_net_gain(f),
                                   load_data(os.path.join(droot, ph, f),
                                             has_header=(dt == "theory")))
                                  for f in files]

all_g = [g for ph in phases for dt in data_types for g, _ in data_by_phase[ph][dt]]
gmin, gmax = min(all_g), max(all_g)

# ── FIGURE & GRID SPEC ───────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 18))
gs  = gridspec.GridSpec(
    7, 3,
    width_ratios=[20, 1.0, 0.6],               # main | spacer | slim bar
    height_ratios=[1.2, 1.2, 1.2, 1.2, 1.2,    # a-e
                   0.45, 0.45],                # f, g
    wspace=0.05, hspace=0.25
)

# (a) phase diagram ---------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0:2])
create_amplitude_plot(ax_a)
fig.add_subplot(gs[0, 2]).axis("off")   # keep width alignment

# (b-e) colour-plots --------------------------------------------------------
ax_b = fig.add_subplot(gs[1, 0:2])
ax_c = fig.add_subplot(gs[2, 0:2])
ax_d = fig.add_subplot(gs[3, 0:2])
ax_e = fig.add_subplot(gs[4, 0:2])

m_b = colourplot(data_by_phase["hermitian"]["experiment"],
                 ax_b, "hermitian", "experiment",
                 gmin, gmax)
m_c = colourplot(data_by_phase["hermitian"]["theory"],
                 ax_c, "hermitian", "theory",
                 gmin, gmax)
m_d = colourplot(data_by_phase["nonhermitian"]["experiment"],
                 ax_d, "nonhermitian", "experiment",
                 gmin, gmax)
m_e = colourplot(data_by_phase["nonhermitian"]["theory"],
                 ax_e, "nonhermitian", "theory",
                 gmin, gmax, label_bottom=True)

# (f, g) line-plots ---------------------------------------------------------
ax_f = fig.add_subplot(gs[5, 0:2])
ax_g = fig.add_subplot(gs[6, 0:2])
fig.add_subplot(gs[5, 2]).axis("off")   # keep spacer & bar columns
fig.add_subplot(gs[6, 2]).axis("off")

create_bottom_panel_hermitian_phase(
    data_by_phase["hermitian"]["experiment"],
    data_by_phase["hermitian"]["theory"],
    ax_f)
create_bottom_panel(
    data_by_phase["nonhermitian"]["experiment"],
    data_by_phase["nonhermitian"]["theory"],
    ax_g)

# move phi-annotations **inside** f & g
for ax, phi in zip([ax_f, ax_g], [r"$\phi = 0$", r"$\phi = \pi$"]):
    ax.set_title("")   # in case helper set one
    ax.text(0.55, 0.88, phi, transform=ax.transAxes,
            ha="left", va="top", fontsize=20)

ax_g.set_xlabel(r"Net Gain, $\Delta G$ [dB]", fontsize=25)

# ── SHARED, SLIMMER COLOUR-BARS -------------------------------------------
cax_bc = fig.add_subplot(gs[1:3, 2])    # rows b & c
cax_de = fig.add_subplot(gs[3:5, 2])    # rows d & e

cb_bc = fig.colorbar(m_c, cax=cax_bc)
cb_bc.set_label(r"$S_{21}$ [dB]", fontsize=20, labelpad=8)
cb_bc.ax.tick_params(labelsize=20)

cb_de = fig.colorbar(m_e, cax=cax_de)
cb_de.set_label(r"$S_{21}$ [dB]", fontsize=20, labelpad=8)
cb_de.ax.tick_params(labelsize=20)

# ── PANEL LETTERS & TICK DENSITY ------------------------------------------
letters = ["a", "b", "c", "d", "e"]
axes    = [ax_a, ax_b, ax_c, ax_d, ax_e]
for lt, ax in zip(letters, axes):
    ax.text(-0.17, 1.05, rf"$\textbf{{{lt}}}$",
            transform=ax.transAxes,
            fontsize=30, fontweight="bold",
            ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis='both', which='major', labelsize=20)

letters = ["f", "g"]
axes    = [ax_f, ax_g]
for lt, ax in zip(letters, axes):
    ax.text(-0.17, 1.44, rf"$\textbf{{{lt}}}$",
            transform=ax.transAxes,
            fontsize=30, fontweight="bold",
            ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis='both', which='major', labelsize=20)    

for a in [ax_f, ax_g]:
    a.yaxis.set_major_locator(MaxNLocator(3))

axes    = [ax_b, ax_c, ax_d, ax_e]
for ax in axes:
    ax.set_ylabel(r"Frequency [GHz]", fontsize=20)
   
# ── SAVE -------------------------------------------------------------------
plt.tight_layout()
fig.subplots_adjust(top=0.94)

out_dir = Path(cwd).parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(out_dir / f"Fig_2.{ext}",
                bbox_inches="tight", pad_inches=0.1, dpi=400)

plt.close()
print("✓ Fig 2 saved – compact layout, slim bars, internal φ-labels.")
