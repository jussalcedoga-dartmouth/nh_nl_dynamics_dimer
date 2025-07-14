#!/usr/bin/env python3
"""
Runs the S-parameter sweep and stores every spectrum to CSV files:

spectra/
 ├─ hermitian/          (phi = 0)
 │   ├─ S21/gain_+8.40dB.csv
 │   └─ S12/…
 └─ nonhermitian/       (phi = pi)
     ├─ S21/…
     └─ S12/…
Each CSV has columns:  frequency_GHz ,  S_dB
"""

import os, json, numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt

# ─────────────────── style (unchanged) ──────────────────────────────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─────────────────── parameters ─────────────────────────────────────────
with open("params.json", "r") as f:
    p = json.load(f)

omega1, omega2           = p["omega1"] * 1e9,       p["omega2"] * 1e9
kappa_drive, kappa_read  = p["kappa_drive"] * 1e6,  p["kappa_readout"] * 1e6
kappa_int_1, kappa_int_2 = p["kappa_int_1"] * 1e6,  p["kappa_int_2"] * 1e6
kappa_c, J0              = p["kappa_c"]   * 1e6,    p["J0"]          * 1e6

kappa_T_1 = kappa_int_1 + kappa_drive + kappa_c
kappa_T_2 = kappa_int_2 + kappa_read  + kappa_c

h_bar, epsilon_dBm = 1.054571817e-34, -30

# ─────────────────── helper functions ───────────────────────────────────
def kappa_T(J_val, kappa_0):          # nonlinear diag term
    return 2 * kappa_0 - J_val

def coupling(phi):                    # coherent hopping
    return 1j * J0 * np.cos(phi/2) * np.exp(1j*phi/2)

P_sat, b_amp = 0.9981e-3, 8.6e-3
alpha_sat = P_sat / (h_bar * omega1 * kappa_c)

def J_nl(alpha, gain):
    if alpha <= alpha_sat:
        return kappa_c * 10**(gain/20)
    num = b_amp + h_bar*omega2*alpha_sat*kappa_c
    den = b_amp + h_bar*omega2*alpha      *kappa_c
    return kappa_c * 10**(gain/20) * (num/den)

def eom(t, y, omega_d, phi, gain, drive):
    a1_r,a1_i,a2_r,a2_i = y
    a1,a2 = a1_r+1j*a1_i , a2_r+1j*a2_i

    eps_W = 10**((epsilon_dBm-30)/10)
    eps   = np.sqrt((kappa_drive*eps_W)/(h_bar*omega_d))

    N1,N2 = (a1.real**2+a1.imag**2),(a2.real**2+a2.imag**2)
    J1,J2 = J_nl(N1,gain), J_nl(N2,gain)
    k1,k2 = kappa_T(J1,kappa_T_1), kappa_T(J2,kappa_T_2)

    eps1 = eps if drive==1 else 0.0
    eps2 = eps if drive==2 else 0.0

    da1 = -(1j*(omega1-omega_d)+k1)*a1 - (1j*J1+coupling(phi))*np.exp(-1j*phi)*a2 + eps1
    da2 = -(1j*(omega2-omega_d)+k2)*a2 - (1j*J2+coupling(phi))*a1                 + eps2
    return [da1.real, da1.imag, da2.real, da2.imag]

def solve_point(job):
    omega_d, gain, phi, drive, read = job
    y0, t_span = [1e7,0,1e7,0], (0.0, 10000.0/kappa_c)

    sol = solve_ivp(eom, t_span, y0,
                    args=(omega_d,phi,gain,drive),
                    dense_output=True, atol=1e-6, rtol=1e-3)

    t_eval = np.linspace(*t_span, 100_000)
    out    = sol.sol(t_eval)
    idx_r,idx_i = ((0,1) if read==1 else (2,3))
    alpha = out[idx_r] + 1j*out[idx_i]
    alpha = alpha[int(0.2*len(alpha)):]          # drop transient

    A0 = np.fft.fft(alpha)[0]/len(alpha)
    photons = A0.real**2 + A0.imag**2
    k_out,w_out = ((kappa_drive,omega1) if read==1 else (kappa_read,omega2))
    P_W = photons*k_out*h_bar*w_out
    S_dB = 10*np.log10(P_W*1e3) - epsilon_dBm
    return omega_d, gain, phi, drive, read, S_dB

# ─────────────────── sweep grids ────────────────────────────────────────
gains       = np.linspace(-4.6, 8.4, 31)
frequencies = np.linspace(5.975e9, 6.085e9, 1000)
phi_vals    = [0.0, np.pi]
cases       = [("S21",1,2), ("S12",2,1)]     # label, drive, readout

cube = np.zeros((2,2,len(frequencies),len(gains)))  # for plotting

jobs = [(f,g,phi,drv,rd) for phi in phi_vals
                                for _,drv,rd in cases
                                for g in gains
                                for f in frequencies]

# ─────────────────── run ODEs ───────────────────────────────────────────
print("Running ODE sweeps …")
with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    for res in tqdm(pool.map(solve_point, jobs), total=len(jobs)):
        omega_d, gain, phi, drv, rd, S = res
        phi_i   = 0 if phi==0.0 else 1
        case_i  = 0 if drv==1   else 1
        f_i     = np.searchsorted(frequencies, omega_d)
        g_i     = np.searchsorted(gains,       gain)
        cube[phi_i,case_i,f_i,g_i] = S

# ─────────────────── save spectra to CSV ────────────────────────────────
base_dir = "spectra"
config_dir = {0:"hermitian", 1:"nonhermitian"}
case_dir  = {0:"S21",        1:"S12"}

for phi_i in range(2):
    for case_i in range(2):
        out_root = os.path.join(base_dir, config_dir[phi_i], case_dir[case_i])
        os.makedirs(out_root, exist_ok=True)
        for g_i, gain in enumerate(gains):
            df = pd.DataFrame({
                "frequency_GHz": frequencies/1e9,
                "S_dB": cube[phi_i,case_i,:,g_i]
            })
            fname = f"gain_{gain:+.2f}dB.csv"
            df.to_csv(os.path.join(out_root, fname), index=False)

print("CSV files written to ./spectra/ …")

# ─────────────────── (optional) plot immediately ───────────────────────
# comment the next two lines if you only want data dumps
from plot_from_csv import plot_from_saved
plot_from_saved("spectra", gains, frequencies)   # quick check
