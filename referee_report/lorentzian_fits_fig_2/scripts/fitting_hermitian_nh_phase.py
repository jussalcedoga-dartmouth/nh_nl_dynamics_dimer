#!/usr/bin/env python3
"""
Robust S21 fitter – TWO Lorentzians (φ = 0) or single Lorentzian (φ = π).
If only one peak is found in the Hermitian window we mirror it.
"""

# ─── imports ─────────────────────────────────────────────────────────────
import glob, pathlib, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy.optimize     import curve_fit, OptimizeWarning
from   scipy.signal       import savgol_filter, peak_widths, find_peaks
from   scipy.interpolate  import CubicSpline
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ─── paths ───────────────────────────────────────────────────────────────
BASE = pathlib.Path(
    "/jumbo/fitzlab/code/nl_nh_dimer/fresh_start/referee_report/lorentzian_fits_fig_2"
)
EXPT, THEO = BASE/"data"/"experiment", BASE/"data"/"theory"
OUT        = BASE/"plots_unified"
(OUT/"summary").mkdir(parents=True, exist_ok=True)
for ph in ["hermitian", "nonhermitian"]:
    (OUT/ph).mkdir(exist_ok=True)

# ─── constants & helpers ────────────────────────────────────────────────
DB, WIN_H, WIN_NH = 20.0, (5.98, 6.075), (6.020, 6.036)
DENSIFY, BIFURC   = 1,    4.78
RGX_EXP = re.compile(r"net_gain_(-?\d+\.?\d*)")
RGX_TH  = re.compile(r"results_gain_(-?\d+\.?\d*)")

lin2db = lambda x: DB*np.log10(np.clip(x, 1e-15, None))
db2lin = lambda d: 10**(d/DB)
lnerr  = lambda x, dx: dx/np.clip(x, 1e-15, None)

# lineshape models
lor  = lambda f, A, f0, g: A*((g/2)**2)/((f-f0)**2+(g/2)**2)
dlor = lambda f, A1, f1, g1, A2, f2, g2, m, C: (
           lor(f, A1, f1, g1) + lor(f, A2, f2, g2) + m*(f-0.5*(f1+f2)) + C
)
slor = lambda f, A, f0, g, C: lor(f, A, f0, g) + C

def safe_fit(model, f, y, p0, bounds):
    lo, hi = bounds
    p = np.clip(p0, lo, hi)
    for _ in range(4):
        try:
            po, pc = curve_fit(model, f, y, p0=p, bounds=bounds, maxfev=80000)
            return po, np.sqrt(np.diag(pc))
        except (RuntimeError, ValueError):
            p = lo + (hi - lo) * np.random.rand(len(p))
    po, pc = curve_fit(model, f, y, p0=None, bounds=bounds, maxfev=80000)
    return po, np.sqrt(np.diag(pc))

def load_csv(path, is_exp, phase):
    if is_exp:
        df = pd.read_csv(path, header=None, names=["f", "p"])
    else:
        df = pd.read_csv(path, comment="#")
        fc = next(c for c in df.columns if "freq"  in c.lower())
        pc = next(c for c in df.columns if "power" in c.lower())
        df = df[[fc, pc]].rename(columns={fc: "f", pc: "p"})
    df.f, df.p = pd.to_numeric(df.f, errors="coerce"), pd.to_numeric(df.p, errors="coerce")
    df = df.dropna()
    if df.f.max() > 1e7:
        df.f /= 1e9
    wl, wh = WIN_H if phase == "hermitian" else WIN_NH
    df = df[(df.f >= wl) & (df.f <= wh)]

    s21d  = df.p.values + (20 if is_exp else 0)
    s21l  = db2lin(s21d)
    f     = df.f.values
    if phase == "nonhermitian" and DENSIFY > 1 and len(f) > 10:
        cs  = CubicSpline(f, s21l)
        fhi = np.linspace(f.min(), f.max(), len(f)*DENSIFY)
        s21l, s21d, f = cs(fhi), lin2db(cs(fhi)), fhi
    return f, s21d, s21l

def half_width(f, y_db, idx):
    w = peak_widths(y_db, [idx], 3/DB)[0][0]
    return max(0.0005, w * np.mean(np.diff(f)))

def seed_double(f, y_db, y_lin):
    mid = 0.5 * (f.min() + f.max())
    sm  = savgol_filter(y_db, 51, 3)
    pk, _ = find_peaks(sm, prominence=1)

    if len(pk) < 2:
        idx    = int(pk[0]) if len(pk) else np.argmax(sm)
        mirror = np.argmin(np.abs(f - (2*mid - f[idx])))
        pk     = np.sort([idx, mirror])
    else:
        left  = pk[f[pk] <= mid]
        right = pk[f[pk] >= mid]
        if len(left) == 0:
            left = np.array([pk[np.argmax(sm[pk])]])
        if len(right) == 0:
            right = np.array([pk[np.argmax(sm[pk])]])
        pk = np.array([left[np.argmax(sm[left])],
                       right[np.argmax(sm[right])]])

    A1, A2 = y_lin[pk] - y_lin.min()
    g1, g2 = half_width(f, y_db, pk[0]), half_width(f, y_db, pk[1])
    return [A1, f[pk[0]], g1, A2, f[pk[1]], g2, 0.0, y_lin.min()]

def fit_trace(f, yL, yD, phase, gain):
    if phase == "hermitian":
        p0 = seed_double(f, yD, yL)
        span = yL.max() - yL.min()
        mid  = 0.5 * (f.min() + f.max())
        loA  = span * 0.05
        bounds = (
            [loA, f.min(), 0.0005, loA, mid, 0.0005, -1e3, yL.min()-span],
            [span*2, mid, 0.3, span*2, f.max(), 0.3, 1e3, yL.min()+span],
        )
        return (*safe_fit(dlor, f, yL, p0, bounds), dlor)

    # π-phase
    f0 = f[np.argmax(yL)]
    A0 = yL.max() - yL.min()
    C0 = yL.min()
    if gain < BIFURC:
        b = ([0, WIN_NH[0], 0.0005, 0],
             [np.inf, WIN_NH[1], 0.02, np.inf])
    else:
        b = ([0, WIN_NH[0], 0.00005, 0],
             [db2lin(15), WIN_NH[1], 0.005, np.inf])
    return (*safe_fit(slor, f, yL, [A0, f0, 0.002, C0], b), slor)

def add_row(table, g, side, A, eA, γ, eγ, C, ph, ds):
    table.append(dict(
        gain=g, phase=ph, side=side, dataset=ds,
        amp_dB=lin2db(A + C),
        err_amp_dB=lin2db(A + C + eA) - lin2db(A + C),
        ln_fwhm=np.log(γ*1e3),
        dln_fwhm=lnerr(γ*1e3, eγ*1e3),
    ))

# ─── processing loop ─────────────────────────────────────────────────────
PNG, REC = [], []
for root, is_exp, rx, label, cd, cf in [
        (EXPT, True,  RGX_EXP, "exp",    "k",   "C0"),
        (THEO, False, RGX_TH,  "theory", "0.5", "C1")]:
    for phase in ["hermitian", "nonhermitian"]:
        for fn in glob.glob(str(root/phase/"*.csv")):
            m = rx.search(fn)
            gain = float(m.group(1)) if m else None
            if gain is None:
                continue

            f, sdb, sln = load_csv(pathlib.Path(fn), is_exp, phase)
            popt, perr, model = fit_trace(f, sln, sdb, phase, gain)
            PNG.append((phase, gain, f, sdb, popt, model, label, cd, cf))

            if phase == "nonhermitian":
                A, f0, γ, C = popt
                eA, eγ      = perr[0], perr[2]
                add_row(REC, gain, "single", A, eA, γ, eγ, C, phase, label)
            else:
                A1, f1, g1, A2, f2, g2, m, C = popt
                eA1, eA2, eγ1, eγ2 = perr[0], perr[3], perr[2], perr[5]
                add_row(REC, gain, "left",  A1, eA1, g1, eγ1, C, phase, label)
                add_row(REC, gain, "right", A2, eA2, g2, eγ2, C, phase, label)

# ─── per-trace PNGs ──────────────────────────────────────────────────────
for ph, gain, f, sdb, popt, model, lab, cd, cf in PNG:
    wl, wh = WIN_H if ph == "hermitian" else WIN_NH
    fd = np.linspace(wl, wh, 4000)

    plt.figure(figsize=(6, 4))

    if ph == 'hermitian':
        plt.plot(f, sdb, ".", color=cd, ms=2, label=f"{lab} data")
        plt.plot(fd, lin2db(model(fd, *popt)), "-", color=cf, lw=1.5, label=f"{lab} fit")
        plt.ylim(-50, np.max(sdb) + 5)
    else:
        plt.plot(f, sdb, ".", color=cd, ms=2, label=f"{lab} data")
        plt.plot(fd, lin2db(model(fd, *popt)), "-", color=cf, lw=1.5, label=f"{lab} fit")

    if ph == "hermitian":
        A1, f1, g1, A2, f2, g2, _, C = popt
        plt.plot(fd, lin2db(lor(fd, A1, f1, g1) + C), "--", color="crimson", lw=1)
        plt.plot(fd, lin2db(lor(fd, A2, f2, g2) + C), "--", color="navy",    lw=1)
        # plt.ylim(-50, max(lin2db(lor(fd, A1, f1, g1) + C))+5)

    plt.xlabel("Frequency [GHz]")
    plt.ylabel("|S21| [dB]")
    plt.title(f"{ph}  ΔG={gain:+.2f} dB")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT/ph/f"gain_{gain:+.2f}_{lab}.png", dpi=300)
    plt.close()

# ─── CSV & summary plots ─────────────────────────────────────────────────
df = pd.DataFrame(REC)
df.to_csv(OUT/"summary"/"all_results.csv", index=False)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(hspace=0.3)
)

sym  = {"exp": "o", "theory": "s"}
fill = {"exp": True, "theory": False}
col  = {"left": "crimson", "right": "navy"}

# Hermitian panel
for side in ["left", "right"]:
    for ds in ["exp", "theory"]:
        sub = df[
            (df["phase"] == "hermitian")
            & (df["side"] == side)
            & (df["dataset"] == ds)
        ]
        if sub.empty:
            continue
        
        ax1.errorbar(
            sub.gain, sub.amp_dB, yerr=sub.err_amp_dB,
            fmt=sym[ds], ms=6,
            mfc=(col[side] if fill[ds] else "none"),
            mec=col[side], ecolor=col[side],
            capsize=3, label=f"{side} ({ds})"
        )

ax1.set_ylabel(r"$S_{21}^{\max}$ [dB]")
ax1.set_title(r"Hermitian ($\phi=0$)")
ax1.legend(fontsize=7, ncol=2)

# Non-Hermitian panel – amplitude
for ds in ["exp", "theory"]:
    sub = df[
        (df["phase"] == "nonhermitian")
        & (df["dataset"] == ds)
    ]
    if sub.empty:
        continue
    ax2.errorbar(
        sub.gain, sub.amp_dB, yerr=sub.err_amp_dB,
        fmt=sym[ds], ms=6,
        mfc=("crimson" if fill[ds] else "none"),
        mec="crimson", ecolor="crimson",
        capsize=3, label=f"amp ({ds})"
    )

ax2.set_xlabel("Net Gain, ΔG [dB]")
ax2.set_ylabel(r"$S_{21}^{\max}$ [dB]", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.legend(fontsize=7, loc="upper left")

# Non-Hermitian panel – linewidth (twin axis)
ax2b = ax2.twinx()
for ds in ["exp", "theory"]:
    sub = df[
        (df["phase"] == "nonhermitian")
        & (df["dataset"] == ds)
    ]
    if sub.empty:
        continue
    ax2b.errorbar(
        sub.gain, sub.ln_fwhm, yerr=sub.dln_fwhm,
        fmt=sym[ds], ms=6,
        mfc=("dodgerblue" if fill[ds] else "none"),
        mec="dodgerblue", ecolor="dodgerblue",
        capsize=3, label=f"ln FWHM ({ds})"
    )

ax2b.set_ylabel("ln FWHM [MHz]", color="dodgerblue")
ax2b.tick_params(axis="y", labelcolor="dodgerblue")
ax2b.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig(OUT/"summary"/"summary_expt_vs_theory.png", dpi=300)
print("✓ All outputs written to", OUT)
