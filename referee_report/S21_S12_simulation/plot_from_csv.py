#!/usr/bin/env python3
"""
Re-loads spectra from ./spectra/** CSV files and recreates the 4-panel figure.
• CSVs with <100 rows or all-zero data are skipped.
• Gain columns that remain all-zero are removed to avoid blank stripes.
"""

import os, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                # headless backend
import matplotlib.pyplot as plt

# ─── plotting style (matches original) ──────────────────────────────────
plt.rcParams.update({
    "font.size": 25,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True
})

def _parse_gain(fname: str) -> float:
    """gain_m4.60dB.csv → -4.6 ; gain_p8.40dB.csv → +8.4"""
    tag = os.path.basename(fname).split("gain_")[1].split("dB")[0]
    return float(tag)          # filenames already encode sign (m/p)

def plot_from_saved(base_dir: str = "spectra",
                    gains: np.ndarray | None = None,
                    freqs: np.ndarray | None = None) -> None:
    cfg_dirs  = ["hermitian", "nonhermitian"]     # φ = 0, π
    case_dirs = ["S21", "S12"]

    # ── infer frequency & gain grids from first good file ───────────────
    if gains is None or freqs is None:
        probe = glob.glob(os.path.join(base_dir,
                                       "hermitian", "S21", "*.csv"))[0]
        gains = sorted(_parse_gain(f) for f in
                       glob.glob(os.path.dirname(probe) + "/*.csv"))
        freqs = pd.read_csv(probe)["frequency_GHz"].to_numpy() * 1e9

    n_f, n_g = len(freqs), len(gains)
    cube = np.zeros((2, 2, n_f, n_g))

    # ── load CSV files with sanity checks ───────────────────────────────
    for phi_i, cfg in enumerate(cfg_dirs):
        for case_i, case in enumerate(case_dirs):
            for csv in glob.glob(os.path.join(base_dir, cfg, case,
                                              "gain_*dB.csv")):
                g_val = _parse_gain(csv)
                if g_val not in gains:
                    continue
                df = pd.read_csv(csv)
                if len(df) < 100:
                    print(f"[skip] {csv}  (only {len(df)} rows)")
                    continue
                data = df["S_dB"].to_numpy()
                if not data.any():              # all zeros
                    print(f"[skip] {csv}  (all-zero data)")
                    continue
                if len(data) != n_f:
                    print(f"[skip] {csv}  (length {len(data)} ≠ {n_f})")
                    continue
                g_idx = np.searchsorted(gains, g_val)
                cube[phi_i, case_i, :, g_idx] = data

    # ── remove gain columns that are still all zeros (“stripes”) ────────
    keep = [i for i in range(n_g) if cube[:, :, :, i].any()]
    if not keep:
        raise RuntimeError("No valid data found in ./spectra/")
    cube      = cube[:, :, :, keep]
    gains_arr = np.array(gains)[keep]

    # ── reproduce original plotting exactly ─────────────────────────────
    freq_GHz   = freqs / 1e9
    mask       = (freq_GHz >= 5.975) & (freq_GHz <= 6.085)
    freq_slice = freq_GHz[mask]
    xticks     = [6.00, 6.03, 6.06]
    phi_ann    = [r"$\phi = 0$", r"$\phi = \pi$"]
    cb_lbl     = [r"$S_{21}\ \mathrm{[dB]}$", r"$S_{12}\ \mathrm{[dB]}$"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11),
                             sharex=True, sharey=True)

    for i_phi in range(2):
        for i_case in range(2):
            ax = axes[i_phi, i_case]
            panel = cube[i_phi, i_case, mask, :].T
            im = ax.imshow(panel, origin="lower", aspect="auto",
                           extent=[freq_slice[0], freq_slice[-1],
                                   gains_arr[0], gains_arr[-1]],
                           cmap="inferno", vmin=-39 if i_phi else None)
            cb = plt.colorbar(im, ax=ax, pad=0.02)
            cb.set_label(cb_lbl[i_case])

            if i_phi == 1:
                ax.set_xlabel("Frequency [GHz]", fontsize=30)
            if i_case == 0:
                ax.set_ylabel(r"Net Gain, $\Delta G$ [dB]", fontsize=30)
            ax.set_xticks(xticks)
            ax.text(0.3, 0.1, phi_ann[i_phi],
                    transform=ax.transAxes,
                    ha="right", va="top",
                    color="white", fontsize=25)

    for lt, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.05, rf"$\textbf{{{lt}}}$",
                transform=ax.transAxes,
                fontsize=30, fontweight="bold",
                ha="right", va="top")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(wspace=0.25, hspace=0.05, top=0.95, bottom=0.08, left=0.12, right=0.92)

    fig.savefig("S_matrix_phase_non_reciprocity.png", dpi=300)
    plt.close()
    print("Plot regenerated → S_matrix_maps_from_csv.png")


# allow command-line use
if __name__ == "__main__":
    plot_from_saved()
