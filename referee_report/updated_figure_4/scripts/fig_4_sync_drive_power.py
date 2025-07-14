import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib import cm, ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable          # noqa: F401
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.ticker as mticker

from get_theory_contours import create_contour_plot
from get_theory_zorro    import get_th_zorro_plot

# ─────────── absolute roots (EDIT ONCE if folders move) ───────────────────
PROJ_ROOT = Path("/jumbo/fitzlab/code/nl_nh_dimer/fresh_start").resolve()

DATA_ROOT = PROJ_ROOT / "figures_paper_def/fig_4_with_theory/data"          # contours & meshes
CSV_DIR   = PROJ_ROOT / "referee_report/updated_figure_4/data"              # drive-power CSVs
PLOT_DIR  = PROJ_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXP_CONT_DIR  = DATA_ROOT / "experiment/contours"
EXP_ZORRO_DIR = DATA_ROOT / "experiment/zorro_plots/synchronization_data"

# ─────────── matplotlib global style ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})
helv = fm.FontProperties(family="Helvetica", weight="bold")

# ─────────── colour maps, labels, helpers (unchanged) ─────────────────────
EPSILONS = ['None', '0', '4', '8', '12', '16']
COLORS   = cm.inferno_r(np.linspace(0, 1, len(EPSILONS)))
TRACE_COL = {0: "darkblue", 4: "crimson"}          # colours for power traces

# # Define the figure and the gridspec layout with a larger figure size
fig = plt.figure(constrained_layout=True, figsize=(24, 13))  # Set the figure size

# ── figure grid : 2 × 6 ──────────────────────────────────────────────
# widths:  (0) inset-exp | (1) c | (2) d | (3) inset-num | (4) e | (5) f
#           └───  left block (3 col)  ───┘ └───  right block (3 col)  ───┘
gs = gridspec.GridSpec(
    2, 7, figure=fig,
    height_ratios=[5.0, 2.0],
    # last entry (column-6) is the colour-bar 
    width_ratios =[1.5, 2.4, 2.4, 1.5, 2.4, 2.4, 0.15],
    wspace=0.0001, hspace=0.03
)

# ── top row ----------------------------------------------------------
ax_large_panel_exp = fig.add_subplot(gs[0, 0:3])     # columns 0-1-2
ax_large_panel_th  = fig.add_subplot(gs[0, 3:6])     # columns 3-4-5

# ── bottom row : experiment -----------------------------------------
ax_inset_exp   = fig.add_subplot(gs[1, 0])           # inset
ax_zorro_exp_0 = fig.add_subplot(gs[1, 1])           # panel c
ax_zorro_exp_4 = fig.add_subplot(gs[1, 2])           # panel d

# ── bottom row : numerics -------------------------------------------
ax_inset_num   = fig.add_subplot(gs[1, 3])           # inset
ax_zorro_th_0  = fig.add_subplot(gs[1, 4])           # panel e
ax_zorro_th_4  = fig.add_subplot(gs[1, 5])           # panel f

# Get the current positions of the top panels
pos_exp = ax_large_panel_exp.get_position()
pos_th = ax_large_panel_th.get_position()

# Adjust positions to make the panels wider while keeping the height and vertical position unchanged
# Increase width by 10% while keeping the vertical position and height unchanged
ax_large_panel_exp.set_position([pos_exp.x0 - 0.05, pos_exp.y0 + 0.155, pos_exp.width, pos_exp.height])
ax_large_panel_th.set_position([pos_th.x0 + 0.05, pos_th.y0 + 0.155, pos_th.width, pos_th.height])

ax_large_panel_th = create_contour_plot(ax_large_panel_th)
ax_zorro_th_0_canvas = get_th_zorro_plot(ax_zorro_th_0, epsilon_dBm=0)
ax_zorro_th_4 = get_th_zorro_plot(ax_zorro_th_4, epsilon_dBm=4)

labels = ['experiment', 'numerics']

# Apply annotations to the big panel
for ax, label in zip([ax_large_panel_exp, ax_large_panel_th], labels):
    ax.text(0.04, 0.95, label, transform=ax.transAxes, fontsize=30,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', edgecolor='none'))

labels = ['experiment', 'numerics'] * 2
for ax, label in zip([ax_zorro_exp_0, ax_zorro_th_0, ax_zorro_exp_4, ax_zorro_th_4], labels):
    if label == 'experiment':
        ax.text(0.60, 0.95, label, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', edgecolor='none'))
    else:
        ax.text(0.66, 0.95, label, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', edgecolor='none'))

for idx, epsilon in enumerate(EPSILONS):
    csv_filename = EXP_CONT_DIR / f"combs_characterization_epsilon_{epsilon}.csv"
    
    # Load data from CSV
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    phases = data[:, 0]
    
    # Read the attenuation headers correctly
    with open(csv_filename, 'r') as file:
        headers = file.readline().strip().split(',')
        attenuations = np.array([float(h) for h in headers[1:]])

    imshow_data = data[:, 1:]

    # Check dimensions
    if len(attenuations) != imshow_data.shape[1]:
        raise ValueError(f"Mismatch in the number of attenuations ({len(attenuations)}) and the number of columns in imshow_data ({imshow_data.shape[1]})")

    # Plotting on the left panel
    contour = ax_large_panel_exp.contourf(attenuations, phases, imshow_data, levels=[0.5, 1], colors=[COLORS[idx]], alpha=0.6, origin='lower')

# Custom legend for the left panel
legend_handles = [Patch(facecolor=COLORS[i], edgecolor='none', label=f'{epsilon} dBm' if epsilon != 'None' else 'None') for i, epsilon in enumerate(EPSILONS)]
ax_large_panel_exp.legend(handles=legend_handles, title=r"Drive Power", loc='lower left', fontsize=20, title_fontsize=20)

def find_nearest_index(array, value):
    index = np.abs(array - value).argmin()
    return index

# Define the function to load and plot the forward sweep data
def load_and_plot_forward(ax, phase, lo_power):
    csv_path = EXP_ZORRO_DIR / f"{phase}_LO_power_{lo_power}_intensity_mesh.csv"

    df = pd.read_csv(csv_path)
    intensity_mesh = df.values
    freq_mesh = df.columns.astype(float)  # assuming the columns are frequency values in GHz

    im = ax.imshow(intensity_mesh, aspect='auto', origin='lower',
                   extent=[freq_mesh[0], freq_mesh[-1], freq_mesh[0], freq_mesh[-1]],
                   interpolation='nearest', cmap='inferno')
    
    if lo_power == '0':
        ax.plot(freq_mesh, freq_mesh, color='darkblue', ls='--', markeredgecolor='white', lw=3.0,  markeredgewidth=1.5)
    else:
        ax.plot(freq_mesh, freq_mesh, color='crimson', ls='--', markeredgecolor='white', lw=3.0,  markeredgewidth=1.5)
    return im

# # Create subplots for forward and backward data in the first column on the right
im_forward_0_exp = load_and_plot_forward(ax_zorro_exp_0, 'nonhermitian', '0')
im_forward_4_exp = load_and_plot_forward(ax_zorro_exp_4, 'nonhermitian', '4')


def set_yaxis_ticks(ax, tick_size=45):
    # Set major locator to MultipleLocator (pi)
    ax.set_ylim(0, 2*np.pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    
    # Define custom formatter to handle specific label formatting
    def format_func(value, pos):
        if np.isclose(value, 0):
            return '$0$'
        elif np.isclose(value, np.pi):
            return r'$\pi$'  # Display 'π' instead of '1π'
        elif np.isclose(value, 2 * np.pi):
            return r'2$\pi$'  # Optionally handle the 2π case as well
        else:
            return f'{value/np.pi:.1g}$\pi$'  # General case

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.tick_params(axis='y', which='major', labelsize=tick_size)

for ax in [ax_large_panel_exp]:
    ax.tick_params(axis='both', which='major', labelsize=40)
    set_yaxis_ticks(ax)
    ax.set_xlim(4.01, ax.get_xlim()[1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

ax_large_panel_exp.set_xlabel(r'$\Delta G$ [dB]', fontsize=40)
ax_large_panel_exp.set_ylabel(r'$\phi$', fontsize=50)

for ax in [ax_zorro_exp_0, ax_zorro_th_0, ax_zorro_exp_4, ax_zorro_th_4]:
    ax.tick_params(axis='both', which='major', labelsize=30)

for ax in [ax_zorro_exp_0, ax_zorro_exp_4]:
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('Meas. Freq. [GHz]', fontsize=30)

# ONE colour-bar, living in panel f’s slot (gs[1,5])
cax  = fig.add_subplot(gs[1, 6])           # dedicated thin column
cbar = fig.colorbar(im_forward_4_exp, cax=cax)
cbar.set_label("Power [dBm]", fontsize=28)
cbar.ax.tick_params(labelsize=24)

# 3)  Helper to read a drive-power CSV ------------------------------------------------
def load_drive_power_csv(kind: str, lo: int):
    fn = CSV_DIR / f"nonhermitian_LO{lo}_drive_power_{kind}.csv"
    df = pd.read_csv(fn)
    return df["drive_frequency_GHz"].to_numpy(float), df["power_at_peak_dBm"].to_numpy(float)

# 4)  Insert a narrow axis left of a given mesh ---------------------------------------
def draw_inset(ax_target, traces, show_ylabel=False):
    for f, p, col, lab in traces:
        ax_target.plot(p, f, lw=0, marker="o", color=col, label=lab)

    # ax_target.set_xlim(-25, 3)
    ax_target.invert_xaxis()
    ax_target.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_target.tick_params(axis="both", labelsize=30)
    ax_target.set_xlabel("Power at\nDrive Peak [dBm]", labelpad=4, fontsize=30)

    if show_ylabel:
        ax_target.set_ylabel("Drive Freq. [GHz]", fontsize=30)
    else:
        ax_target.tick_params(axis="y", left=False, labelleft=False)

    ax_target.legend(frameon=False, fontsize=17, handlelength=0.1)

# 5)  Experimental inset (left of panel c) -------------------------------------------
# left-hand inset (experiment, panel c)
draw_inset(
    ax_inset_exp,
    [(load_drive_power_csv("exp", lo)[0],
      load_drive_power_csv("exp", lo)[1],
      TRACE_COL[lo], f"{lo} dBm") for lo in (0, 4)],
    show_ylabel=True
)

# numerics inset (right block)
draw_inset(
    ax_inset_num,
    [(load_drive_power_csv("theory", lo)[0],
      load_drive_power_csv("theory", lo)[1],
      TRACE_COL[lo], f"{lo} dBm") for lo in (0, 4)],
    show_ylabel=False
)

for ax in [ax_zorro_exp_0, ax_zorro_exp_4, ax_zorro_th_0, ax_zorro_th_4]:
    # exactly three evenly-spaced tick *positions*
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # keep the tick marks but hide their labels
    ax.yaxis.set_major_formatter(mticker.NullFormatter())   # no text
    ax.tick_params(axis="y", which="major",
                   left=True,      # show the ticks
                   labelleft=False)  # but suppress labels

# # Little annotations...
ax_zorro_exp_0.annotate(r'0 dBm', xy=(0.8, 0.2), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=30, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))
ax_zorro_th_0.annotate(r'0 dBm', xy=(0.8, 0.2), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=30, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))

ax_zorro_exp_4.annotate(r'4 dBm', xy=(0.8, 0.2), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=30, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))
ax_zorro_th_4.annotate(r'4 dBm', xy=(0.8, 0.2), xycoords='axes fraction', ha='center', va='top', color='red', fontsize=30, bbox=dict(boxstyle='round,pad=0.3', fc='none', edgecolor='none'))

# Add labels to each subplot with Helvetica
ax_large_panel_exp.text(-0.08, 1.15, r'$\textbf{a}$',
                        transform=ax_large_panel_exp.transAxes,
                        fontsize=40, fontproperties=helv, va='top', ha='right')
ax_large_panel_th.text(-0.08, 1.15, r'$\textbf{b}$',
                        transform=ax_large_panel_th.transAxes,
                        fontsize=40, fontproperties=helv, va='top', ha='right')

ax_inset_exp.text(-0.05, 1.22, r'$\textbf{c}$',
                    transform=ax_inset_exp.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')
ax_zorro_exp_0.text(-0.05, 1.22, r'$\textbf{d}$',
                    transform=ax_zorro_exp_0.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')
ax_zorro_exp_4.text(-0.05, 1.22, r'$\textbf{e}$',
                    transform=ax_zorro_exp_4.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')

ax_inset_num.text(-0.05, 1.22, r'$\textbf{f}$',
                    transform=ax_inset_num.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')
ax_zorro_th_0.text(-0.05, 1.22, r'$\textbf{g}$',
                    transform=ax_zorro_th_0.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')
ax_zorro_th_4.text(-0.05, 1.22, r'$\textbf{h}$',
                    transform=ax_zorro_th_4.transAxes,
                    fontsize=40, fontproperties=helv, va='top', ha='right')

# save
for ext in ("png", "pdf", "svg"):
    plt.savefig(f"../plots/Fig_4.{ext}",
                bbox_inches="tight", pad_inches=0.1, dpi=400)
print(f"✓ Fig_4 saved → {PLOT_DIR}")