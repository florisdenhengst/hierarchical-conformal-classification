"""What split conformal prediction does, in two panels, for the project page.

(a) calibration: the nonconformity scores of the calibration set and the
    threshold q-hat at their ceil((n+1)(1-alpha))/n empirical quantile.
(b) test input: every class whose nonconformity score falls below q-hat enters
    the prediction set.

Edit the numbers below and re-run:  python talk/fig_conformal.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

# --- palette (Paul Tol bright, as in the paper) ------------------------------
BLUE, RED = "#4477AA", "#EE6677"
INK, MUTED, GRID = "#1A1A1A", "#6E6E6E", "#DDDDDD"
FILL, PALE = "#BBCCEE", "#EEEEEE"

ALPHA = 0.10          # target error level
QHAT = 0.80           # threshold implied by the calibration scores below

# nonconformity scores of a test input, s(x, y) = 1 - f(x)_y, per class
TEST = [("agama", 0.42), ("alligator lizard", 0.55), ("green lizard", 0.61),
        ("whiptail", 0.72), ("gecko", 0.88), ("iguana", 0.93), ("turtle", 0.97)]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw=dict(wspace=0.32))

    # --- (a) calibration ------------------------------------------------------
    rng = np.random.default_rng(0)
    scores = rng.beta(2.0, 3.6, 4000)
    # rescale so the conformal quantile of this sample really is QHAT
    n = len(scores)
    level = np.ceil((n + 1) * (1 - ALPHA)) / n
    scores = scores * QHAT / np.quantile(scores, level)
    scores = scores[scores <= 1.0]
    bins = np.linspace(0, 1, 46)
    counts, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [FILL if c <= QHAT else PALE for c in centers]
    ax1.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9,
            color=colors, edgecolor="none")

    ax1.axvline(QHAT, color=RED, lw=2, zorder=5)
    ax1.annotate(r"$\hat{q}$", xy=(QHAT, ax1.get_ylim()[1] * 0.92),
                 xytext=(6, 0), textcoords="offset points",
                 color=RED, fontsize=15, va="center")
    ax1.text(0.40, ax1.get_ylim()[1] * 0.60,
             f"{int((1 - ALPHA) * 100)}% of\ncalibration scores",
             ha="center", va="center", color=BLUE, fontsize=11)

    ax1.set_title("a · calibrate once", loc="left", fontsize=13, color=INK, pad=12)
    ax1.set_xlabel(r"nonconformity score  $s(x_i, y_i)$", fontsize=11, color=INK)
    ax1.set_ylabel("calibration examples", fontsize=11, color=INK)
    ax1.set_yticks([])
    ax1.set_xlim(0, 1)

    # --- (b) one test input ---------------------------------------------------
    labels = [t[0] for t in TEST][::-1]
    values = [t[1] for t in TEST][::-1]
    ypos = np.arange(len(labels))
    bar_colors = [BLUE if v <= QHAT else "#CCCCCC" for v in values]
    ax2.barh(ypos, values, height=0.62, color=bar_colors, edgecolor="none")

    ax2.axvline(QHAT, color=RED, lw=2, zorder=5)
    ax2.text(QHAT + 0.015, len(labels) - 0.35, r"$\hat{q}$",
             color=RED, fontsize=15, va="center")

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_title("b · predict a set", loc="left", fontsize=13, color=INK, pad=12)
    ax2.set_xlabel(r"nonconformity score  $s(x, y)$", fontsize=11, color=INK)
    ax2.set_xlim(0, 1.06)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, length=3)
        ax.set_axisbelow(True)

    # class names of the selected classes in full ink (after tick_params)
    for tick, v in zip(ax2.get_yticklabels(), values):
        tick.set_color(INK if v <= QHAT else MUTED)

    fig.text(0.5, -0.09,
             r"$C(x) = \{\, y : s(x,y) \leq \hat{q} \,\}$"
             "    contains the true label with probability at least "
             r"$1 - \alpha$",
             ha="center", fontsize=12.5, color=INK)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig6_conformal.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", os.path.join(OUT, "fig6_conformal.png"))


if __name__ == "__main__":
    main()
