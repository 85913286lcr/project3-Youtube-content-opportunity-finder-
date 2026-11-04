"""Shared matplotlib style so every figure in the notebooks looks the same."""
import matplotlib.pyplot as plt

from src import config

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, INK2 = "#b5b4ad", "#0b0b0b", "#52514e"
NICHE_COLORS = {"AI Career": BLUE, "AI Tools": ORANGE, "AI Agents": AQUA}


def use():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GREY, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "lines.linewidth": 2, "font.size": 10, "figure.dpi": 110,
    })


def save(fig, name):
    fig.savefig(config.FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
