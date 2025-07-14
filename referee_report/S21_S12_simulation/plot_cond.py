#!/usr/bin/env python3
"""
Convenience wrapper: import plot_from_csv and regenerate the figure.
Usage:  python regenerate_plot.py
"""

from plot_from_csv import plot_from_saved

if __name__ == "__main__":
    plot_from_saved()          # defaults to ./spectra directory
