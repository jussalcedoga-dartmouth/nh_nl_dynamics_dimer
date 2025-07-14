#!/usr/bin/env python3
"""
make_combined_dimer_emission_fig.py
-----------------------------------
Combine
  ├─ dimer_emission_Pd_exp_minus.png
  └─ dimer_emission_Pd_exp_plus.png
into a single figure with Helvetica labels “a” and “b”.
"""

import matplotlib
matplotlib.use("Agg")                  # comment out if you want an interactive backend
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.image as mpimg
from pathlib import Path

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
IMG_MINUS = Path("../plots/dimer_emission_Pd_exp_minus.png")
IMG_PLUS  = Path("../plots/dimer_emission_Pd_exp_plus.png")
OUTFILE   = Path("../plots/dimer_emission_combined.png")

# Helvetica font properties (fallbacks to default sans-serif if Helvetica not installed)
helv = fm.FontProperties(family="Helvetica", size=15)

# ------------------------------------------------------------------
# LOAD IMAGES
# ------------------------------------------------------------------
img_minus = mpimg.imread(IMG_MINUS)
img_plus  = mpimg.imread(IMG_PLUS)

# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(4, 6))

# left panel (a)
axes[0].imshow(img_minus)
axes[0].axis("off")
axes[0].text(-0.1, 0.98, r"$\textbf{{a}}$",
             transform=axes[0].transAxes,
             fontproperties=helv, ha="left", va="top")

# right panel (b)
axes[1].imshow(img_plus)
axes[1].axis("off")
axes[1].text(-0.1, 0.98, r"$\textbf{{b}}$",
             transform=axes[1].transAxes,
             fontproperties=helv, ha="left", va="top")

plt.tight_layout()
fig.savefig(OUTFILE, dpi=300, bbox_inches="tight")
print(f"✓ Combined figure saved → {OUTFILE.resolve()}")
