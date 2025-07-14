# ───── imports ──────────────────────────────────────────────────────────
import re, sys, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ───── locate ROOT (folder that owns data/experiment) ───────────────────
ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT/'data/experiment').exists():
    ROOT = ROOT.parent
if not (ROOT/'data/experiment').exists():
    sys.exit("❌  cannot find project root with  data/experiment/")

EXP_DIR = ROOT/'data/experiment/nonhermitian'
THR_DIR = ROOT/'data/theory/nonhermitian'
for d in (EXP_DIR, THR_DIR):
    if not d.exists():
        sys.exit(f"❌  missing folder {d}")

# ───── pick csv with largest positive ΔG in name ────────────────────────
GAIN_RX = re.compile(r'([-+]?\d+(?:\.\d+)?)')

def highest_csv(folder: Path):
    best, g_max = None, -1e9
    for csv in folder.glob('*.csv'):
        m = GAIN_RX.search(csv.stem)
        if m:
            g = float(m.group(1))
            if g > g_max:
                best, g_max = csv, g
    if best is None:
        sys.exit(f'❌  no “gain” number in filenames under {folder}')
    print(f'{folder.name:10s} → {best.name}   (ΔG ≈ {g_max:+.2f} dB)')
    return best

CSV_EXP = highest_csv(EXP_DIR)
CSV_THR = highest_csv(THR_DIR)

# ───── simple CSV loader ────────────────────────────────────────────────
DB        = 20.0
WIN_GHZ   = (6.024, 6.033)          # narrow window around resonance
lin2db    = lambda x: DB*np.log10(np.clip(x, 1e-20, None))
db2lin    = lambda d: 10**(d/DB)

def load_csv(fp: Path, add20):
    df = pd.read_csv(fp, header=None, usecols=[0,1])
    f  = pd.to_numeric(df.iloc[:,0], errors='coerce').to_numpy()/1e9   # → GHz
    y  = pd.to_numeric(df.iloc[:,1], errors='coerce').to_numpy()
    if add20: y += 20.0
    m   = np.isfinite(f) & np.isfinite(y) & (f>=WIN_GHZ[0]) & (f<=WIN_GHZ[1])
    return f[m], y[m]                   # y still in dB

f_exp, y_exp = load_csv(CSV_EXP, True)
f_thr, y_thr = load_csv(CSV_THR, False)

# ───── Fano model (linear units) + dB wrapper ───────────────────────────
def fano_lin(f, A, f0, G, q, C):
    eps = 2*(f-f0)/G
    return C * (1.0 + A*(q+eps)**2 / (1.0+eps**2))

def fano_db(f,*p):  return lin2db(fano_lin(f,*p))

# ───── improved initial-guess builder ───────────────────────────────────
def smart_seed(f, y_db):
    y_lin = db2lin(y_db)

    # baseline = median of lowest 25 %
    idx_low  = np.argsort(y_db)[:max(3, int(0.25*len(y_db)))]
    C0_lin   = np.median(y_lin[idx_low])

    # peak
    idx_pk   = y_lin.argmax()
    f0       = f[idx_pk]

    # amplitude to hit the peak exactly
    A0       = max(y_lin[idx_pk]/C0_lin - 1.0, 0.05)
    A0       = np.clip(A0, 0.05, 200.0)

    # width Γ - full width at 30 % contrast
    thr_lin  = C0_lin + 0.30*(y_lin[idx_pk]-C0_lin)
    above    = np.where(y_lin > thr_lin)[0]
    if above.size >= 2:
        G0 = max(f[above[-1]] - f[above[0]], 3e-4)
    else:
        G0 = 6e-4                               # fallback 600 kHz

    # sign of derivative a few points left → q0
    left_idx = max(0, idx_pk-3)
    slope    = np.sign(y_lin[left_idx+1] - y_lin[left_idx])
    q0       =  1.0 if slope > 0 else -1.0

    return np.array([A0, f0, G0, q0, C0_lin])

# ───── bounded least-squares fit ────────────────────────────────────────
def fit_fano(f, y_db):
    p0 = smart_seed(f, y_db)
    lo = np.array([0.02, p0[1]-0.002, 1e-4, -10, p0[4]*0.3])
    hi = np.array([300., p0[1]+0.002, 3e-3,  10, p0[4]*3.0])

    # keep p0 inside bounds
    p0 = np.clip(p0, lo, hi)

    popt,_ = curve_fit(fano_lin, f, db2lin(y_db),
                       p0=p0, bounds=(lo,hi), maxfev=50000)
    return popt

p_exp = fit_fano(f_exp, y_exp)
p_thr = fit_fano(f_thr, y_thr)

# report
for tag,p in [('exp',p_exp),('theory',p_thr)]:
    A,f0,G,q,C = p
    print(f'{tag:6s}:  A={A:7.3f}  f0={f0:.6f} GHz  Γ={G*1e3:6.2f} MHz  '
          f'q={q:+6.2f}   C={lin2db(C):+6.2f} dB')

# ───── plot ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
fd = np.linspace(*WIN_GHZ, 2500)
plt.plot(f_exp, y_exp, 'k.',  ms=3, label='exp data')
plt.plot(f_thr, y_thr, 'crimson.', ms=3, label='theory data')
plt.plot(fd, fano_db(fd,*p_exp), 'tab:blue',   lw=2, label='exp fit')
plt.plot(fd, fano_db(fd,*p_thr), 'tab:orange', lw=2, label='theory fit')
plt.title('Non-Hermitian – highest ΔG  (Fano fits)')
plt.xlabel('Frequency [GHz]'); plt.ylabel(r'$|S_{21}|$  [dB]')
plt.ylim(-30, None); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig('fano_highgain.png', dpi=350)
print('\n✓  figure saved →', Path.cwd()/'fano_highgain.png')
