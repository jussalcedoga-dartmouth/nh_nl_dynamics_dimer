#!/usr/bin/env python3
"""
suppl_mat_fit.py — legends restored, sensible sig-figs, tighter bG bounds
"""

import re, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from glob import glob
import matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, OptimizeWarning
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ── reference targets ───────────────────────────────────────────────────
BG_REF_MW, PSAT_REF_MW = 8.6, 0.9981
BG_REF_W , PSAT_REF_W  = BG_REF_MW*1e-3, PSAT_REF_MW*1e-3

# ── plotting style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 28,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ── unit helpers ────────────────────────────────────────────────────────
dBm2W = lambda d: 10**((d-30)/10)
W2dBm = lambda w: 10*np.log10(w)+30
W2mW  = lambda w: w*1e3

def parse_lo(f):
    m=re.search(r'LO_POWER_(-?\d+(?:\.\d+)?)_?dBm',f,re.I)
    return float(m.group(1)) if m else np.nan

def gain_piecewise(P,y0,bG,Psat):
    return np.where(P<=Psat, y0, y0*(bG+Psat)/(bG+P))

def knee_1dB(pin_dBm,pout_dBm,n_lin=4):
    if pin_dBm.size<=n_lin: return dBm2W(pin_dBm[-1])
    m,c=np.linalg.lstsq(np.vstack([pin_dBm[:n_lin],np.ones(n_lin)]).T,
                        pout_dBm[:n_lin],rcond=None)[0]
    for i in range(n_lin, pin_dBm.size):
        if m*pin_dBm[i]+c-pout_dBm[i]>=1: return dBm2W(pin_dBm[i])
    return dBm2W(pin_dBm[-1])

# pretty value(err) formatter  –  2-sig-fig error, matched decimals
def val_err(value, err, max_dec=6):
    """
    Return 'value(err)' with the error printed to 2 significant digits.
    If the error would round to zero, the decimal precision is increased
    until a non-zero digit appears (capped at max_dec = 6).
    """
    if not np.isfinite(err) or err <= 0:
        return f"{value:.3f}"
    # magnitude of the error
    exp  = int(np.floor(np.log10(err)))
    # two significant digits
    decimals = max(0, -exp + 1)          # 2 sig-fig ⇒ +1 extra digit
    decimals = min(decimals, max_dec)
    err_digits = int(round(err * 10**decimals))
    # if still zero, keep adding decimals (up to max_dec)
    while err_digits == 0 and decimals < max_dec:
        decimals += 1
        err_digits = int(round(err * 10**decimals))
    value_fmt = f"{{:.{decimals}f}}"
    return f"{value_fmt.format(round(value, decimals))}({err_digits})"

def wmean(vals, errs):
    v,e=np.array(vals),np.array(errs)
    m=(np.isfinite(v)&np.isfinite(e)&(e>0))
    if m.sum()==0: return v.mean(),np.nan
    w=1/e[m]**2
    return np.sum(w*v[m])/np.sum(w), np.sqrt(1/np.sum(w))

# ── locate data ─────────────────────────────────────────────────────────
ROOT=Path.cwd()/"G_0_compression"
if not ROOT.is_dir(): sys.exit("  G_0_compression/ not found.")
drives=sorted([d for d in ROOT.glob("omega_d_*_GHz") if d.is_dir()],
              key=lambda p: float(p.name.split('_')[2]))
if not drives: sys.exit("  No omega_d_*_GHz folders.")

plots_root=Path("plots_final"); plots_root.mkdir(exist_ok=True)
out_types=("data_power_out_amp","data_power_out_ampda")
cmap=plt.cm.get_cmap("inferno",len(drives))

proxy_Psat=Line2D([],[],c='k',ls=':',lw=2,label=r"$P_\mathrm{sat}$")
# proxy_y0  =Line2D([],[],c='k',ls=':',lw=2,label=r"$y_0$")

# ── loop over datasets ──────────────────────────────────────────────────
for out_type in out_types:

    fig,(ax_io,ax_lin)=plt.subplots(1,2,figsize=(18,4),gridspec_kw={"wspace":.28})
    ax_io .text(-0.12,1.05,r"$\textbf{a}$",transform=ax_io .transAxes,
                fontsize=40,fontweight="bold")
    ax_lin.text(-0.12,1.05,r"$\textbf{b}$",transform=ax_lin.transAxes,
                fontsize=40,fontweight="bold")

    psat_vals,psat_errs,bg_vals,bg_errs=[],[],[],[]

    for idx,ddir in enumerate(drives):
        colour=cmap(idx)
        freq=float(ddir.name.split('_')[2])
        label=rf"{freq:.2f} GHz"

        in_dir,out_dir=ddir/"data_power_into_amp", ddir/out_type
        if not(in_dir.is_dir() and out_dir.is_dir()): continue

        files_in =sorted(glob(str(in_dir /"*.csv")))
        files_out=sorted(glob(str(out_dir/"*.csv")))
        Pin_map={parse_lo(f): pd.read_csv(f)["trace_corrected_dBm"].max()
                 for f in files_in if not np.isnan(parse_lo(f))}

        Pin_dBm,Pout_dBm=[],[]
        for f in files_out:
            lo=parse_lo(f)
            if np.isnan(lo) or lo not in Pin_map: continue
            Pin_dBm.append(Pin_map[lo]); Pout_dBm.append(
                pd.read_csv(f)["trace_corrected_dBm"].max())
        if not Pin_dBm: continue

        o=np.argsort(Pin_dBm)
        Pin_dBm,Pout_dBm=np.array(Pin_dBm)[o],np.array(Pout_dBm)[o]
        G0_dB=Pout_dBm-Pin_dBm; G0_lin=10**(G0_dB/20)
        Pin_W=dBm2W(Pin_dBm); Pin_mW=W2mW(Pin_W)

        P_knee=knee_1dB(Pin_dBm,Pout_dBm)
        bounds=([0,0.8*BG_REF_W,0.7*P_knee],[np.inf,1.5*BG_REF_W,1.3*P_knee])
        p0=[G0_lin.max(), BG_REF_W, P_knee]

        try:
            popt,pcov=curve_fit(gain_piecewise,Pin_W,G0_lin,p0=p0,
                                bounds=bounds,maxfev=3e5)
            y0,bG,Psat=popt
            _,bG_err,Psat_err=np.sqrt(np.diag(pcov))
        except OptimizeWarning:
            continue

        psat_vals.append(W2mW(Psat));  psat_errs.append(W2mW(Psat_err))
        bg_vals  .append(W2mW(bG));    bg_errs  .append(W2mW(bG_err))

        # plotting with legend label
        ax_io.scatter(Pin_dBm,Pout_dBm,c=[colour],s=70,label=label)
        xg=np.linspace(Pin_W.min(),Pin_W.max(),400)
        ax_lin.scatter(Pin_mW,G0_lin,c=[colour],s=70,alpha=.45)
        ax_lin.plot(W2mW(xg),gain_piecewise(xg,y0,bG,Psat),
                    c=colour,lw=3,ls='--')
        ax_lin.axvline(W2mW(Psat),c=colour,ls=':',lw=2)
        # ax_lin.axhline(y0,c=colour,ls=':',lw=2)

    # dataset-level weighted means
    psat_mean,psat_sig=wmean(psat_vals,psat_errs)
    bg_mean  ,bg_sig  =wmean(bg_vals ,bg_errs )

    print(f"== {out_type:<18} ==>  "
          f"Psat = {val_err(psat_mean,psat_sig)} mW   |   "
          f"bG = {val_err(bg_mean,bg_sig)} mW")

    # axis / legend
    ax_io .set_xlabel("Input Power [dBm]"); ax_io .set_ylabel("Output Power [dBm]")
    ax_lin.set_xlabel("Input Power [mW]");  ax_lin.set_ylabel(r"$10^{G_0/20}$")
    ax_lin.set_xscale("log")
    ax_io.legend(fontsize=18)
    h,l=ax_io.get_legend_handles_labels()
    # h+=[proxy_Psat,proxy_y0]; l+=[r"$P_\mathrm{sat}$",r"$y_0$"]
    ax_lin.legend(h,l,fontsize=18)

    out_dir=plots_root/out_type; out_dir.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf","svg"):
        fig.savefig(out_dir/f"amp_digital_att.{ext}",bbox_inches="tight",dpi=400)
    plt.close(fig)
