from IPython import embed
import pylab as plt

plt.rc('text', usetex=True)

 
custom_preamble = {
            "text.usetex": True,
            "font.size" : 10,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bbold} \usepackage{amssymb} \usepackage{mathrsfs} \usepackage{physics} \usepackage{mathtools}"}
plt.rcParams.update(custom_preamble)

import numpy as np
from qutip import *


from matplotlib import cm, ticker, colors
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
from matplotlib.ticker import ScalarFormatter # heeft nooit gewerkt


