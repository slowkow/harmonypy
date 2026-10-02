"""Make the figures from results/ (run the three run_* scripts first).

    uv run --no-project --with pandas --with matplotlib python plot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Palette and chrome from the dataviz reference palette (light mode).
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
LAB_COLORS = {"A": "#b9b7af", "B": VIOLET}

BUILDS = {
    "r_harmony": ("R harmony (harmony2 branch)", INK),
    "v2.0.2": ("harmonypy 2.0.2", ORANGE),
    "pr56": ("harmonypy + PR #56", BLUE),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def toy_scatter():
    toy = pd.read_csv(HERE / "data" / "toy.tsv", sep="\t")
    panels = [("Input", toy[["x", "y"]].values)]
    for key in BUILDS:
        corrected = pd.read_csv(RESULTS / f"toy_corrected_{key}.tsv", sep="\t")
        panels.append((BUILDS[key][0], corrected[["x", "y"]].values))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharex=True, sharey=True)
    for ax, (title, xy) in zip(axes, panels):
        for lab in ["A", "B"]:
            m = toy.lab.values == lab
            ax.scatter(
                xy[m, 0], xy[m, 1], s=16, color=LAB_COLORS[lab],
                edgecolor=SURFACE, linewidth=0.6, label=f"lab {lab}",
                zorder=3 if lab == "B" else 2,
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.8, 7.2)
        ax.set_xlabel("x")
        ax.text(2.5, 6.4, "cell type 2", color=MUTED, fontsize=9)
        ax.text(7.3, 2.3, "cell type 1", color=MUTED, fontsize=9, ha="right")
    axes[0].set_ylabel("y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2)
    axes[0].annotate(
        "lab B: 10 of its\n100 cells here", xy=(1.8, 5.2), xytext=(3.3, 3.9),
        fontsize=9, color=INK2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    fig.suptitle(
        "batch_prop_cutoff = 0.15: lab B's mean assignment to cell type 2 is "
        "0.10, below the cutoff, so lab B should not be corrected there",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_toy_scatter.png")
    plt.close(fig)


def toy_sweep():
    toy = pd.read_csv(HERE / "data" / "toy.tsv", sep="\t")
    in_type2 = toy.cell_type == 2
    gap_input = np.linalg.norm(
        toy.loc[in_type2 & (toy.lab == "A"), ["x", "y"]].mean()
        - toy.loc[in_type2 & (toy.lab == "B"), ["x", "y"]].mean()
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    style = {
        "r_harmony": dict(lw=5, alpha=0.35),
        "v2.0.2": dict(lw=2),
        "pr56": dict(lw=2, ls=(0, (4, 3))),
    }
    for key, (name, color) in BUILDS.items():
        sweep = pd.read_csv(RESULTS / f"toy_sweep_{key}.tsv", sep="\t")
        ax.plot(sweep.cutoff, sweep.gap_type2, color=color, label=name,
                drawstyle="steps-post", **style[key])

    ax.axhline(gap_input, color=MUTED, lw=0.8, ls=":")
    ax.text(0.302, gap_input, "uncorrected", color=MUTED, fontsize=9,
            va="center")
    for x, text in [(0.10, "0.10 = lab B's mean\nassignment to cell type 2"),
                    (0.20, "0.20 = twice that,\nwhat 2.0.2 compared")]:
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x + 0.004, 0.55, text, color=INK2, fontsize=9, va="center")

    ax.text(0.01, 0.12, "R harmony and PR #56\n(lines overlap)", color=INK,
            fontsize=9)
    ax.text(0.152, 0.12, "2.0.2 still\ncorrects lab B", color=ORANGE,
            fontsize=9, fontweight="bold")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("batch_prop_cutoff")
    ax.set_ylabel("Distance between lab A and lab B\nin cell type 2 after correction")
    ax.set_title("Where each implementation stops correcting lab B in cell type 2")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_toy_cutoff_sweep.png")
    plt.close(fig)


def ircolitis_correlation():
    rows = []
    for cutoff in ["1e-05", "0.01"]:
        ref = pd.read_csv(
            RESULTS / f"ircolitis_r_harmony_cutoff{cutoff}.tsv.gz", sep="\t"
        ).values
        for key in ["v2.0.2", "pr56"]:
            z = pd.read_csv(
                RESULTS / f"ircolitis_{key}_cutoff{cutoff}.tsv.gz", sep="\t"
            ).values
            for i in range(ref.shape[1]):
                r = np.corrcoef(z[:, i], ref[:, i])[0, 1]
                rows.append({"cutoff": cutoff, "build": key, "pc": i + 1, "r": r})
    cor = pd.DataFrame(rows)
    cor.to_csv(RESULTS / "ircolitis_correlation_with_r.tsv", sep="\t",
               index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    titles = {
        "1e-05": "batch_prop_cutoff = 1e-5 (default)",
        "0.01": "batch_prop_cutoff = 0.01",
    }
    for ax, cutoff in zip(axes, ["1e-05", "0.01"]):
        sub = cor[cor.cutoff == cutoff]
        v = sub[sub.build == "v2.0.2"]
        p = sub[sub.build == "pr56"]
        ax.scatter(v.pc, v.r, s=64, facecolor="none", edgecolor=ORANGE,
                   linewidth=1.5, label=BUILDS["v2.0.2"][0], zorder=3)
        ax.scatter(p.pc, p.r, s=18, color=BLUE, label=BUILDS["pr56"][0],
                   zorder=4)
        ax.axhline(0.99, color=MUTED, lw=0.8, ls=":")
        ax.set_title(titles[cutoff])
        ax.set_xlabel("Principal component")
        ax.set_xlim(0, 51)
        n_v = int((v.r < 0.99).sum())
        n_p = int((p.r < 0.99).sum())
        note = (
            "Identical output from both builds.  Dotted line: r = 0.99"
            if np.allclose(v.r.values, p.r.values)
            else f"PCs with r < 0.99 (dotted line):  2.0.2: {n_v},  PR #56: {n_p}"
        )
        ax.text(1, 0.9135, note, color=INK2, fontsize=9)
    axes[0].set_ylabel("Pearson r with R harmony, per PC")
    axes[0].set_ylim(0.91, 1.002)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
    fig.suptitle(
        "ircolitis blood CD8 T cells (68,785 cells, 50 PCs), "
        "corrected for donor and batch",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_ircolitis_vs_r.png")
    plt.close(fig)

    summary = cor.groupby(["cutoff", "build"]).r.agg(
        ["min", "median", lambda r: int((r < 0.99).sum())]
    )
    summary.columns = ["min_r", "median_r", "pcs_below_0.99"]
    print(summary.round(4))


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    toy_scatter()
    toy_sweep()
    ircolitis_correlation()
