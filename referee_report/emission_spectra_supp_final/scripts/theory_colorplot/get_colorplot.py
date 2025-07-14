#!/usr/bin/env python3
"""
dimer_emission_colorplots_v4b.py
────────────────────────────────
Exactly your v4 script, but the colour-bar is created with
mpl_toolkits.axes_grid1.make_axes_locatable so it hugs panel “b”
while the large wspace (≈0.20) between panels “a” and “b” is kept.
"""

# ─────────────── imports & MPI config ───────────────────────────────────
import numpy as np, matplotlib, json, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # <- NEW
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 40,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─────────── load numerical parameters (unchanged) ──────────────────────
with open("params.json") as f:
    par = json.load(f)

omega1, omega2 = par["omega1"]*1e9, par["omega2"]*1e9
kappa_drive    = par["kappa_drive"]  *1e6
kappa_readout  = par["kappa_readout"]*1e6
kappa_int_1    = par["kappa_int_1"]  *1e6
kappa_int_2    = par["kappa_int_2"]  *1e6
kappa_c        = par["kappa_c"]      *1e6
J0             = par["J0"]           *1e6
hbar = 1.054_571_817e-34

gain_par = {"b": 8.6e-3, "P_sat": 0.9981e-3}
P_sat, b_amp = gain_par["P_sat"], gain_par["b"]
alpha_sat = P_sat / (hbar*omega1*kappa_c)

kappa_t_1 = kappa_int_1 + kappa_drive   + kappa_c
kappa_t_2 = kappa_int_2 + kappa_readout + kappa_c

def kappa_total(Jnl, k0): return 2*k0 - Jnl
def coupling_phi(phi):   return 1j*J0*np.cos(phi/2)*np.exp(1j*phi/2)

def J_nl(alpha, dg):
    pref = kappa_c*10**(dg/20)
    if alpha <= alpha_sat: return pref
    num = b_amp + hbar*omega2*alpha_sat*kappa_c
    den = b_amp + hbar*omega2*alpha     *kappa_c
    return pref*(num/den)

def rhs(t, y, phi, dg, om_d, drv_dBm):
    a1r,a1i,a2r,a2i = y
    eps = 0
    if drv_dBm is not None:
        drv_W = 10**((drv_dBm-30)/10)
        eps   = np.sqrt((kappa_drive*drv_W)/(hbar*om_d))
    a1,a2 = a1r+1j*a1i, a2r+1j*a2i
    n1,n2 = abs(a1)**2, abs(a2)**2
    J1,J2 = J_nl(n1,dg), J_nl(n2,dg)
    k1,k2 = kappa_total(J1,kappa_t_1), kappa_total(J2,kappa_t_2)
    da1 = -(k1)*a1 - (1j*J1+coupling_phi(phi))*np.exp(-1j*phi)*a2 + eps
    da2 = -(k2)*a2 - (1j*J2+coupling_phi(phi))*a1
    return [da1.real,da1.imag,da2.real,da2.imag]

def simulate(task):
    dg, phi, P = task
    y0=[1e7,0,1e7,0]
    t_span=(0,10000/kappa_c)
    om_d=omega1
    sol=solve_ivp(rhs,t_span,y0,args=(phi,dg,om_d,P),
                  atol=1e-6,rtol=1e-3,dense_output=True)
    t=np.linspace(*t_span,10000)
    a2=sol.sol(t)[2]+1j*sol.sol(t)[3]
    t,a2=t[int(0.2*len(t)):],a2[int(0.2*len(t)):]
    fft=np.fft.fft(a2)
    freq=np.fft.fftfreq(len(a2),d=t[1]-t[0])*2*np.pi
    psd=(fft/len(a2)).real**2+(fft/len(a2)).imag**2
    freq,psd=np.fft.fftshift(freq),np.fft.fftshift(psd)
    p_W=psd*kappa_readout*hbar*omega2
    p_dBm=10*np.log10(p_W*1e3)
    freq_GHz = om_d/1e9 + freq/(2*np.pi*1e9)
    return P,phi,freq_GHz,p_dBm

# ───────────────────── run sweep ─────────────────────────────────────────
delta_gain=8.4
phi_vals=[np.pi-0.5,np.pi+0.5]
drive_vals=np.arange(-10,21,0.02)
tasks=[(delta_gain,phi,P) for phi in phi_vals for P in drive_vals]

with ProcessPoolExecutor(max_workers=min(os.cpu_count(),len(tasks))) as pool:
    res=list(tqdm(pool.map(simulate,tasks),total=len(tasks)))

freq_grid=None
data={phi:[] for phi in phi_vals}
for P,phi,f,psd in res:
    if freq_grid is None: freq_grid=f
    data[phi].append((P,psd))
for phi in phi_vals:
    data[phi].sort(key=lambda x:x[0])
    data[phi] = np.stack([row[1] for row in data[phi]])[:, ::-1]

# ───────────────────── plotting ──────────────────────────────────────────
fig, ax = plt.subplots(1,2,figsize=(22,10),sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.1,left=0.08,right=0.90,
                    top=0.89,bottom=0.15)

extent=[freq_grid.min(),freq_grid.max(),drive_vals.min(),drive_vals.max()]
cmap="inferno"

im_a=ax[0].imshow(data[phi_vals[0]],origin="lower",aspect="auto",
                  extent=extent,cmap=cmap,vmin=-60,interpolation="nearest")
im_b=ax[1].imshow(data[phi_vals[1]],origin="lower",aspect="auto",
                  extent=extent,cmap=cmap,vmin=-60,interpolation="nearest")

for a in ax: a.set_xlabel("Frequency [GHz]")
ax[0].set_ylabel("Drive power [dBm]")
ax[0].set_title(r"$\phi = \pi - 0.5\;(\Delta G = 8.4\,\mathrm{dB})$",pad=14)
ax[1].set_title(r"$\phi = \pi + 0.5\;(\Delta G = 8.4\,\mathrm{dB})$",pad=14)
ax[0].annotate(r"\textbf{a}",xy=(-0.12,1.05),xycoords="axes fraction",
               fontsize=50,fontweight="bold")
ax[1].annotate(r"\textbf{b}",xy=(-0.1,1.05),xycoords="axes fraction",
               fontsize=50,fontweight="bold")

# ───── tight colour-bar next to panel b ──────────────────────────────────
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%", pad=0.1)
cb = fig.colorbar(im_b, cax=cax)
cb.set_label("Power [dBm]")

dash_pattern = [12, 7]          # ← 12-pt dash, 8-pt gap

line = ax[0].axvline(omega1/1e9, color="crimson", lw=3.5)
line.set_dashes(dash_pattern)

line = ax[1].axvline(omega1/1e9, color="crimson", lw=3.5)
line.set_dashes(dash_pattern)

bbox_kw = dict(boxstyle="round,pad=0.4", fc="gray", ec="none")


ax[0].annotate("numerics",
               xy=(0.95, 0.06), xycoords="axes fraction",
               ha="right", va="bottom", fontsize=38, bbox=bbox_kw)


ax[1].annotate("numerics",
               xy=(0.05, 0.06), xycoords="axes fraction",
               ha="left", va="bottom", fontsize=38, bbox=bbox_kw)

fig.savefig("phase_locking_theory.png", dpi=300)
fig.savefig("phase_locking_theory.pdf", dpi=300)

print("✓ Figure saved → phase_locking_theory.png")
