#!/usr/bin/env python3
"""
Fig-2: compact layout with slim colour-bars and internal φ-labels.
Bottom panels f & g are populated directly from
    ../data/amplitude_linewidth_results.csv
(which has columns
    gain,phase,side,dataset,amp_dB,err_amp_dB,log10_fwhm,dlog10_fwhm)
"""

# ─── imports ─────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import pandas as pd

# project-local helpers (phase diagram & colour plots)
from get_plots_individually import *   # panel (a)
from return_phase_diagram import *                         # noqa: F401

# ─── global style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        22,
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Light"],
    "text.usetex":      True,
})

# ─── paths & constants ───────────────────────────────────────────────────
cwd   = os.getcwd()
csv_path = Path(cwd).parent / "data" / "amplitude_linewidth_log10.csv"

# colour-map y-ticks
wanted_freqs = [6.00, 6.03, 6.06]   # GHz

# dashed vertical line (bifurcation / onset of limit cycle)
BIFURC = 4.78                       # dB

# ------------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------------
def nearest_idx(arr, val):
    return np.abs(arr - val).argmin()

# ─── colour-plot factory (unchanged) ─────────────────────────────────────
def colourplot(data, ax, phase_tag, data_type,
               gmin, gmax, label_bottom=False):
    gains = np.array([d[0] for d in data])
    trans = np.array([d[1]["Power_dB"].values for d in data])
    freqs = data[0][1]["Frequency"].values / 1e9          # → GHz

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
    ax.axvline(BIFURC, ls="--", lw=3, color="white", alpha=0.5)
    phi = r"$\phi = 0$" if phase_tag == "hermitian" else r"$\phi = \pi$"
    ax.text(0.20, 0.20, phi, transform=ax.transAxes,
            ha="right", va="top", color="white", fontsize=25)
    tag = "experiment" if data_type == "experiment" else "numerics"
    ax.text(0.05, 0.90, tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="none", alpha=0.8))
    return m

def populate_bottom_panels(csv_file, ax_herm, ax_nh):
    """
    Panel f (Hermitian, φ = 0) and Panel g (Non-Hermitian, φ = π)

    • Hermitian:  left/right peaks → crimson/navy
                  experiment = filled circles, numerics = open squares
    • Non-Hermitian: red (amp) + blue (ln FWHM) with same marker code
    """

    df = pd.read_csv(csv_file)

    # mapping from dataset tag → legend text
    ds_label = {"exp": "exp", "theory": "num"}

    sym  = {"exp": "o",  "theory": "s"}          # marker shapes
    fill = {"exp": True, "theory": False}        # filled vs open
    lw   = {"exp": 2.0,  "theory": 1.0}          # error-bar line widths
    cap  = {"exp": 4,    "theory": 3}            # cap sizes

    # ── Hermitian (φ = 0) --------------------------------------------------
    herm  = df[df.phase == "hermitian"]
    color = {"left": "crimson", "right": "navy"}

    for side in ["left", "right"]:
        for ds in ["exp", "theory"]:
            sub = herm[(herm.side == side) & (herm.dataset == ds)]
            if sub.empty:
                continue
            ax_herm.errorbar(
                sub.gain, sub.amp_dB, yerr=sub.err_amp_dB,
                fmt=sym[ds], ms=5,
                mfc=(color[side] if fill[ds] else "none"),
                mec=color[side], ecolor=color[side],
                elinewidth=lw[ds], capsize=cap[ds],
                alpha=(0.6 if ds == "exp" else 1.0),
                label=f"{side.capitalize()} ({ds_label[ds]})"
            )

    ax_herm.axvline(BIFURC, ls="--", color="black", lw=2)
    ax_herm.set_ylabel(r"$S_{21}^{\max}$ [dB]", color="black")
    ax_herm.tick_params(axis="y", labelcolor="black")
    ax_herm.set_xlim(df.gain.min(), df.gain.max())
    ax_herm.legend(fontsize=12, ncol=2, loc="upper left")

    # ── Non-Hermitian (φ = π) ---------------------------------------------
    nh = df[df.phase == "nonhermitian"]

    # amplitude (red) – left y-axis
    for ds in ["exp", "theory"]:
        sub = nh[nh.dataset == ds]
        if sub.empty:
            continue
        ax_nh.errorbar(
            sub.gain, sub.amp_dB, yerr=sub.err_amp_dB,
            fmt=sym[ds], ms=5,
            mfc=("crimson" if fill[ds] else "none"),
            mec="crimson", ecolor="crimson",
            elinewidth=lw[ds], capsize=cap[ds],
            alpha=(0.6 if ds == "exp" else 1.0),
            label=rf"$S_{{21}}^{{\max}}$ ({ds_label[ds]})"
        )

    ax_nh.set_ylabel(r"$S_{21}^{\max}$ [dB]", color="crimson")
    ax_nh.tick_params(axis="y", labelcolor="crimson")
    ax_nh.set_xlim(df.gain.min(), df.gain.max())

    # linewidth (blue) – right y-axis
    ax_nh2 = ax_nh.twinx()
    for ds in ["exp", "theory"]:
        sub = nh[nh.dataset == ds]
        if sub.empty:
            continue
        ax_nh2.errorbar(
            sub.gain, sub.log10_fwhm, yerr=sub.dlog10_fwhm,
            fmt=sym[ds], ms=5,
            mfc=("darkblue" if fill[ds] else "none"),
            mec="darkblue", ecolor="darkblue",
            elinewidth=lw[ds], capsize=cap[ds],
            alpha=(0.6 if ds == "exp" else 1.0),
            label=rf"$\log_{{10}}$ FWHM ({ds_label[ds]})"
        )

    ax_nh2.set_ylabel(r"$\log_{10}$ FWHM [MHz]", color="darkblue")
    ax_nh2.tick_params(axis="y", labelcolor="darkblue")

    # ── combined legend ----------------------------------------------------
    handles_amp,  labels_amp  = ax_nh.get_legend_handles_labels()
    handles_fwhm, labels_fwhm = ax_nh2.get_legend_handles_labels()
    handles = handles_amp + handles_fwhm
    labels  = labels_amp  + labels_fwhm

    ax_nh.axvline(BIFURC, ls="--", color="black", lw=2.0)

    ax_nh.legend(
        handles, labels,
        fontsize=12, ncol=2,
        columnspacing=0.8, handletextpad=0.2, labelspacing=0.2,
        loc="upper left", bbox_to_anchor=(0.0, 0.83, 0.0, 0.0),
        framealpha=0.9
    )

    if ax_nh2.legend_ is not None:
        ax_nh2.legend_.remove()

# ─── colour-plot data (unchanged)  ───────────────────────────────────────
#   We still need these for sub-plots b–e
data_base_path  = Path(cwd).parent / "data"
experiment_path = data_base_path / "experiment"
theory_path     = data_base_path / "theory"

def load_data(fp, has_header):
    df = pd.read_csv(fp) if has_header else pd.read_csv(
        fp, header=None, names=["Frequency", "Power_dB"])
    if not has_header:
        df["Power_dB"] += 20     # external attenuation in dB
    return df

def extract_net_gain(fname):
    return float(fname.split("_")[-1].replace(".csv", ""))

phases     = ["hermitian", "nonhermitian"]
data_types = ["experiment", "theory"]

data_by_phase = {}
for ph in phases:
    data_by_phase[ph] = {}
    for dt in data_types:
        droot = experiment_path if dt == "experiment" else theory_path
        files = sorted(os.listdir(droot / ph), key=extract_net_gain)
        data_by_phase[ph][dt] = [
            (extract_net_gain(f),
             load_data(droot / ph / f, has_header=(dt == "theory")))
            for f in files
        ]

all_g = [g for ph in phases for dt in data_types
         for g, _ in data_by_phase[ph][dt]]
gmin, gmax = min(all_g), max(all_g)

# ─── figure & GridSpec ───────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 18.5))
gs  = gridspec.GridSpec(
    7, 3,
    width_ratios=[20, 1.0, 0.6],        # main | spacer | slim bar
    height_ratios=[0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8],
    wspace=0.05, hspace=0.25
)

# (a) phase diagram --------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0:2])
create_amplitude_plot(ax_a)
fig.add_subplot(gs[0, 2]).axis("off")   # spacer keeps width alignment

# (b-e) colour maps --------------------------------------------------------
ax_b = fig.add_subplot(gs[1, 0:2])
ax_c = fig.add_subplot(gs[2, 0:2])
ax_d = fig.add_subplot(gs[3, 0:2])
ax_e = fig.add_subplot(gs[4, 0:2])

m_b = colourplot(data_by_phase["hermitian"]["experiment"],
                 ax_b, "hermitian", "experiment", gmin, gmax)
m_c = colourplot(data_by_phase["hermitian"]["theory"],
                 ax_c, "hermitian", "theory",     gmin, gmax)
m_d = colourplot(data_by_phase["nonhermitian"]["experiment"],
                 ax_d, "nonhermitian", "experiment", gmin, gmax)
m_e = colourplot(data_by_phase["nonhermitian"]["theory"],
                 ax_e, "nonhermitian", "theory",     gmin, gmax)

# (f, g) bottom panels -----------------------------------------------------
ax_f = fig.add_subplot(gs[5, 0:2])
ax_g = fig.add_subplot(gs[6, 0:2])
fig.add_subplot(gs[5, 2]).axis("off")   # keep spacer / colour-bar columns
fig.add_subplot(gs[6, 2]).axis("off")

populate_bottom_panels(csv_path, ax_f, ax_g)


# move φ annotations inside f & g
for ax, phi in zip([ax_f], [r"$\phi = 0$"]):
    ax.text(0.85, 0.3, phi, transform=ax.transAxes,
            ha="left", va="top", fontsize=20)
    
for ax, phi in zip([ax_g], [r"$\phi = \pi$"]):
    ax.text(0.85, 0.65, phi, transform=ax.transAxes,
            ha="left", va="top", fontsize=20)

ax_g.set_xlabel(r"Net Gain, $\Delta G$ [dB]", fontsize=25)

# ─── shared slim colour-bars ---------------------------------------------
cax_bc = fig.add_subplot(gs[1:3, 2])     # rows b & c
cax_de = fig.add_subplot(gs[3:5, 2])     # rows d & e

cb_bc = fig.colorbar(m_c, cax=cax_bc)
cb_bc.set_label(r"$S_{21}$ [dB]", fontsize=20, labelpad=8)
cb_bc.ax.tick_params(labelsize=20)

cb_de = fig.colorbar(m_e, cax=cax_de)
cb_de.set_label(r"$S_{21}$ [dB]", fontsize=20, labelpad=8)
cb_de.ax.tick_params(labelsize=20)

# ─── panel letters & tick density  ----------------------------------------
letters = list("abcde")
for lt, ax in zip(letters, [ax_a, ax_b, ax_c, ax_d, ax_e]):
    ax.text(-0.17, 1.05, rf"$\textbf{{{lt}}}$",
            transform=ax.transAxes, fontsize=30, fontweight="bold",
            ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="both", labelsize=20)

for lt, ax in zip("fg", [ax_f, ax_g]):
    ax.text(-0.17, 1.44, rf"$\textbf{{{lt}}}$",
            transform=ax.transAxes, fontsize=30, fontweight="bold",
            ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="both", labelsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(3))

# y-labels for colour maps
for ax in [ax_b, ax_c, ax_d, ax_e]:
    ax.set_ylabel(r"Frequency [GHz]", fontsize=20)

# ─── save -----------------------------------------------------------------
plt.tight_layout()
fig.subplots_adjust(top=0.94)

out_dir = Path(cwd).parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(out_dir / f"Fig_2.{ext}",
                bbox_inches="tight", pad_inches=0.1, dpi=400)

plt.close()
print("✓ Fig 2 saved – bottom panels now use amplitude_linewidth_results.csv")
