"""Consistent chart styling for all notebooks (colour-blind-safe categorical order)."""
import matplotlib.pyplot as plt

# Validated categorical order: slot 1 = normal / primary series, slot 2 = suspicious / comparison.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
NORMAL, SUSPICIOUS = BLUE, ORANGE
MUTED = "#8a8984"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.prop_cycle": plt.cycler(color=SERIES), "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 9,
    })
