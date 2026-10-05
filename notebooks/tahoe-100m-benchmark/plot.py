#!/usr/bin/env python
"""Draw the benchmark figure and a summary from results.jsonl.

usage: plot.py [--results results.jsonl] [--out benchmark-2026-10]

Writes OUT.png, OUT.pdf and OUT-summary.md. harmonypy 0.2.0 numbers come from
the April 2026 benchmark on the same machine (harmonypy-0.2-april-2026.json
next to this script, or the file named by the HISTORY environment variable).
The agreement section of the summary needs the corrected coordinates that
run.sh saves in Z/; without them it is left out. Every number and
claim on the figure is computed from the data, and a statement whose data are
missing is left out, so the figure can be drawn while runs are still going.
Set NEXT_LABEL (e.g. "2.1.0") once the next version number is chosen.
"""
import argparse
import json
import os
import statistics
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.environ.get("HISTORY", os.path.join(HERE, "harmonypy-0.2-april-2026.json"))
MAIN, OLD = "main-623ff51", "2.0.2"
NEXT = os.environ.get("NEXT_LABEL", "next release")
COMMIT = MAIN.split("-")[-1]
CELL_SETS = ["1M", "2M", "4M", "8M", "16M", "full"]
SAMPLE_SETS = ["50B", "100B", "200B", "400B", "800B"]

# Line colors keep the old figure's grey (0.2) and teal (2.0); the next release is Okabe-Ito vermillion.
# Text uses darker shades of the same hues (contrast >= 4.5:1 on white).
COLOR = {"0.2": "#999999", OLD: "#00BFC4", MAIN: "#D55E00"}
TEXT = {"0.2": "#737373", OLD: "#008084", MAIN: "#B84F00"}
STYLE = {
    "0.2": dict(color=COLOR["0.2"], ls="--", marker="s", ms=5, lw=1.6,
                label="harmonypy 0.2 (PyTorch, April 2026)"),
    OLD: dict(color=COLOR[OLD], ls="-", marker="D", ms=5, lw=2.0, label="harmonypy 2.0.2 (current release)"),
    MAIN: dict(color=COLOR[MAIN], ls="-", marker="o", ms=6, lw=2.4, zorder=3,
               label=f"harmonypy {NEXT} (next release)" if NEXT[0].isdigit() else "harmonypy next release"),
}
TIME_TICKS = {1: "1 s", 10: "10 s", 60: "1 min", 600: "10 min", 3600: "1 h", 4 * 3600: "4 h", 24 * 3600: "1 day"}
GB_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.labelsize": 10.5})


def threads_used(r):
    """Threads main runs with: ncores, capped at the CPUs it may use (all of them by default)."""
    if r["build"] != MAIN:
        return None
    return min(r["ncores"], r["available_cpus"]) if r["ncores"] else r["available_cpus"]


def config_key(r):
    """Runs with the same key are repeats of one configuration."""
    return (r["build"], r["dataset"], threads_used(r), json.dumps(r.get("kwargs", {}), sort_keys=True), r["cpus"])


def load_runs(path):
    """Group repeats; returns (configurations, failed records)."""
    groups, failed = defaultdict(list), []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (failed if r.get("failed") else groups[config_key(r)]).append(r)
    configs = {}
    for key, rs in groups.items():
        walls = [r["wall_seconds"] for r in rs]
        configs[key] = dict(
            n_cells=rs[0]["n_cells"],
            n_batches=rs[0]["n_batches"],
            available_cpus=rs[0]["available_cpus"],
            wall=statistics.median(walls),
            wall_min=min(walls),
            wall_max=max(walls),
            cpp=statistics.median(r["cpp_seconds"] for r in rs),
            run=statistics.median(r["run_harmony_seconds"] for r in rs),
            extract=statistics.median(r["extract_seconds"] for r in rs),
            cpu=statistics.median(r["cpu_seconds"] for r in rs),
            others=statistics.median(r["others_cpu_seconds"] for r in rs),
            peak=statistics.median(r["peak_rss_gb"] for r in rs),
            loaded=statistics.median(r["rss_after_loading_gb"] for r in rs),
            load_seconds=statistics.median(r["load_seconds"] for r in rs),
            iterations=sorted({r["iterations"] for r in rs}),
            repeats=len(rs),
            sha=sorted({r["extension_sha256"] for r in rs}),
        )
    return configs, failed


def default(configs, build, dataset):
    """The configuration a user gets: default settings, every CPU of the machine."""
    most = max((c["available_cpus"] for c in configs.values()), default=0)
    for (b, d, threads, kwargs, cpus), v in configs.items():
        if b == build and d == dataset and kwargs == "{}" and v["available_cpus"] == most \
                and threads in (None, most):
            return v
    return None


def history():
    """harmonypy 0.2.0 from April 2026. Its memory is what the process held after the run, not a peak."""
    if not os.path.exists(HISTORY):
        return {}
    d = json.load(open(HISTORY))
    return {
        "batches": [(r["n_batches"], r["time_seconds"]) for r in d.get("harmony1_batch", [])],
        "cells": [(r["n_cells"], r["time_seconds"]) for r in d.get("harmony1_cell", [])],
        "cells_gb": [(r["n_cells"], r["rss_total_gb"]) for r in d.get("harmony1_cell", [])],
    }


def iteration_note(cells, samples, full):
    """'every run converged after 1 Harmony iteration', naming any exceptions."""
    runs = [(b, d, v) for table in (cells, samples) for b, pts in table.items() for d, v in pts.items()]
    counts = defaultdict(int)
    for _, _, v in runs:
        for i in v["iterations"]:
            counts[i] += 1
    if not counts:
        return None
    usual = max(counts, key=counts.get)
    plural = lambda n: f"{n} Harmony iteration{'s' if n != 1 else ''}"
    exceptions = [(b, d, v) for b, d, v in runs if v["iterations"] != [usual]]
    note = f"every run converged after {plural(usual)}"
    if exceptions:
        names = {OLD: "2.0.2", MAIN: f"the {NEXT}"}
        where = lambda d: "all cells" if d == full else (f"{d:g}M cells" if isinstance(d, float) else f"{d} samples")
        note += " except " + ", ".join(f"{names[b]} on {where(d)} ({'/'.join(map(str, v['iterations']))})"
                                       for b, d, v in exceptions)
    return note


def fmt_time(s):
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min" if s < 600 else f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def time_axis(ax, label="Run time (log scale)"):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(list(TIME_TICKS)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: TIME_TICKS.get(round(v), "")))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(label)
    ax.grid(True, which="major", axis="y", color="#e5e5e5", lw=0.8)


def plain_log_x(ax, ticks, labels, lo, hi, base=10):
    ax.set_xscale("log", base=base)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, i: dict(zip(ticks, labels)).get(v, "")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lo, hi)


def label(ax, x, y, text, build, dx=7, dy=0, ha="left", va="center", **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=kw.pop("fontsize", 9.5), color=TEXT[build], **kw)


def speedup(ax, x, y_slow, y_fast, color="#333333", dx=-8):
    """A two-headed arrow from y_slow down to y_fast at x, with 'N× faster' beside it."""
    ax.annotate("", xy=(x, y_fast), xytext=(x, y_slow),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1, shrinkA=6, shrinkB=6))
    ax.annotate(f"{y_slow / y_fast:.0f}×\nfaster", (x, np.sqrt(y_slow * y_fast)), xytext=(dx, 0),
                textcoords="offset points", ha="right" if dx < 0 else "left", va="center",
                fontsize=10.5, fontweight="bold", color=color, linespacing=1.0)


def plot_cells(ax, xs, ys, build, full, **extra):
    """The 1-16M subsamples as one line; all cells (1,344 samples, not 800) joined by a dotted segment."""
    style = dict(STYLE[build], **extra)
    sub = [(x, y) for x, y in zip(xs, ys) if x != full]
    if sub:
        ax.plot([x for x, _ in sub], [y for _, y in sub], **style)
    if full in xs:
        y_full = ys[xs.index(full)]
        if sub:
            ax.plot([sub[-1][0], full], [sub[-1][1], y_full], color=style["color"], ls=":",
                    lw=style["lw"] * 0.7, alpha=style.get("alpha", 1.0))
        point = {k: v for k, v in style.items() if k not in ("ls", "label", "lw")}
        ax.plot([full], [y_full], ls="none", **point)


def agreement(z_dir):
    """Per dataset: minimum per-PC correlation and relative RMS difference, main vs 2.0.2."""
    out = {}
    for d in ("1M", "16M"):
        a, b = os.path.join(z_dir, f"main-{d}.npy"), os.path.join(z_dir, f"2.0.2-{d}.npy")
        if os.path.exists(a) and os.path.exists(b):
            x, y = np.load(a).astype(np.float64), np.load(b).astype(np.float64)
            r = min(float(np.corrcoef(x[:, j], y[:, j])[0, 1]) for j in range(x.shape[1]))
            rel = float(np.linalg.norm(x - y) / np.linalg.norm(y - y.mean(axis=0)))
            out[d] = (r, rel)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.jsonl")
    p.add_argument("--out", default="benchmark-2026-10")
    a = p.parse_args()
    configs, failed = load_runs(a.results)
    hist = history()
    full = next((v["n_cells"] for k, v in configs.items() if k[1] == "full"), 95_596_109) / 1e6
    full_batches = next((v["n_batches"] for k, v in configs.items() if k[1] == "full"), 1344)
    x_of = {d: (full if d == "full" else float(d[:-1])) for d in CELL_SETS}
    cells = {b: {x_of[d]: v for d in CELL_SETS if (v := default(configs, b, d))} for b in (OLD, MAIN)}
    samples = {b: {int(d[:-1]): v for d in SAMPLE_SETS if (v := default(configs, b, d))} for b in (OLD, MAIN)}
    most = max((c["available_cpus"] for c in configs.values()), default=0)
    threads = {}
    for d in ("16M", "1M"):
        pts = {k[2]: v for k, v in configs.items()
               if k[0] == MAIN and k[1] == d and k[3] == "{}" and v["available_cpus"] == most}
        threads[d] = sorted(pts.items())

    fig = plt.figure(figsize=(11, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], left=0.075, right=0.985, top=0.815, bottom=0.17,
                          hspace=0.42, wspace=0.17)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[1, 1])
    x_cells = [1, 2, 4, 8, 16, full]
    x_cells_labels = ["1", "2", "4", "8", "16", f"{full:.1f}\nall cells"]

    # (a) Run time against the number of cells.
    ax = ax_a
    if hist.get("cells"):
        xs, ys = zip(*hist["cells"])
        ax.plot(np.array(xs) / 1e6, ys, **STYLE["0.2"])
        label(ax, xs[-1] / 1e6, ys[-1], fmt_time(ys[-1]), "0.2")
    for build in (OLD, MAIN):
        if cells[build]:
            plot_cells(ax, list(cells[build]), [v["wall"] for v in cells[build].values()], build, full)
    for build in (OLD, MAIN):
        if cells[build]:
            x = max(cells[build])
            v, other = cells[build][x], cells[MAIN if build == OLD else OLD].get(x)
            label(ax, x, v["wall"], fmt_time(v["wall"]), build, fontweight="bold")
            if other and v["iterations"] != other["iterations"]:
                its = "/".join(map(str, v["iterations"]))
                label(ax, x, v["wall"], f"{its} iteration{'s' if its != '1' else ''}", build, dy=-12, fontsize=8.5)
    if 16 in cells[MAIN] and max(cells[MAIN]) != 16:
        label(ax, 16, cells[MAIN][16]["wall"], fmt_time(cells[MAIN][16]["wall"]), MAIN, dx=0, dy=-9, ha="center",
              va="top")
    shared = sorted(set(cells[OLD]) & set(cells[MAIN]))
    if shared:
        x = shared[-1]
        speedup(ax, x, cells[OLD][x]["wall"], cells[MAIN][x]["wall"])
    plain_log_x(ax, x_cells, x_cells_labels, 0.75, 260)
    ax.set_xlabel("Number of cells (millions, log scale)")
    time_axis(ax)
    ax.set_title("Run time by number of cells")

    # (b) Run time against the number of samples, 1M cells.
    ax = ax_b
    if hist.get("batches"):
        xs, ys = zip(*hist["batches"])
        ax.plot(xs, ys, **STYLE["0.2"])
        label(ax, xs[-1], ys[-1], fmt_time(ys[-1]), "0.2")
    for build in (OLD, MAIN):
        if samples[build]:
            xs, vs = list(samples[build]), list(samples[build].values())
            ax.plot(xs, [v["wall"] for v in vs], **STYLE[build])
            label(ax, xs[-1], vs[-1]["wall"], fmt_time(vs[-1]["wall"]), build,
                  fontweight="bold" if build == MAIN else None)
    plain_log_x(ax, [50, 100, 200, 400, 800], ["50", "100", "200", "400", "800"], 38, 2400, base=2)
    ax.set_xlabel("Number of samples (batches), log scale")
    ax.tick_params(labelleft=False)
    ax.grid(True, which="major", axis="y", color="#e5e5e5", lw=0.8)
    ax.set_title("Run time by number of samples (1M cells)")

    # (c) Peak memory against the number of cells.
    ax = ax_c
    if hist.get("cells_gb"):
        xs, ys = zip(*hist["cells_gb"])
        ax.plot(np.array(xs) / 1e6, ys, **STYLE["0.2"])
        label(ax, xs[-1] / 1e6, ys[-1], f"{ys[-1]:.0f} GB", "0.2")
    # Where the builds' memory nearly overlaps, 2.0.2 is drawn wide and pale under the next release.
    overlap = any(abs(np.log(cells[OLD][x]["peak"] / cells[MAIN][x]["peak"])) < np.log(1.1)
                  for x in set(cells[OLD]) & set(cells[MAIN]))
    for build in (OLD, MAIN):
        if cells[build]:
            extra = {"lw": 5, "alpha": 0.55, "ms": 7} if build == OLD and overlap else {}
            plot_cells(ax, list(cells[build]), [v["peak"] for v in cells[build].values()], build, full, **extra)
    ends = {b: cells[b][max(cells[b])] for b in (OLD, MAIN) if cells[b]}
    for build, v in ends.items():
        x = max(cells[build])
        other = ends.get(OLD if build == MAIN else MAIN)
        close = other is not None and abs(np.log(v["peak"] / other["peak"])) < np.log(1.5)
        dy = (7 if v["peak"] >= other["peak"] else -7) if close else 0
        label(ax, x, v["peak"], f"{v['peak']:.0f} GB", build, dy=dy, fontweight="bold" if build == MAIN else None)
    sub = {x: v for x, v in cells[MAIN].items() if x != full}
    if len(sub) >= 2:
        xs = np.array(list(sub))
        slope = float(np.polyfit(xs, np.array([v["peak"] for v in sub.values()]), 1)[0])  # GB per million cells
        mid = xs[len(xs) // 2]
        name = NEXT if NEXT[0].isdigit() else "next release"
        ax.annotate(f"{name}: about {slope:.1f} GB\nper million cells", (mid, sub[mid]["peak"]), xytext=(12, -14),
                    textcoords="offset points", ha="left", va="top", fontsize=9.5, color=TEXT[MAIN])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(GB_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of cells (millions, log scale)")
    ax.set_ylabel("Peak memory (GB, log scale)")
    ax.grid(True, which="major", axis="y", color="#e5e5e5", lw=0.8)
    ax.set_title("Peak memory by number of cells")
    ax.tick_params(labelbottom=True)

    # (d) Run time against the number of threads: next release on 16M and 1M cells; 2.0.2 uses one thread.
    ax = ax_d
    ax.axvline(64, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
    ax.annotate("64 cores; more\nthreads share them", (64, 0.6), xycoords=("data", "axes fraction"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, color="#777777")
    for d, alpha in (("16M", 1.0), ("1M", 0.55)):
        if not threads[d]:
            continue
        ns = np.array([n for n, _ in threads[d]])
        ws = np.array([v["wall"] for _, v in threads[d]])
        ax.plot(ns, ws, **{**STYLE[MAIN], "alpha": alpha})
        # Lines are pale for 1M cells; text stays dark enough to read.
        label(ax, ns[0], ws[0], f"{d} cells: {fmt_time(ws[0])}", MAIN, dx=6, dy=5, ha="left", va="bottom")
        label(ax, ns[-1], ws[-1], fmt_time(ws[-1]), MAIN, fontweight="bold" if d == "16M" else None)
        old = default(configs, OLD, d)
        if old:
            ax.plot([1, ns[-1]], [old["wall"]] * 2, color=COLOR[OLD], ls=(0, (4, 3)), lw=1.6, alpha=alpha)
            ax.plot([1], [old["wall"]], color=COLOR[OLD], marker="D", ms=5, alpha=alpha)
            label(ax, ns[-1], old["wall"], f"2.0.2, {d}: {fmt_time(old['wall'])}", OLD)
    plain_log_x(ax, [1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"], 0.8, 700, base=2)
    ax.set_xlabel("Threads (ncores; the default is all 128)")
    time_axis(ax)
    old_runs = [v for k, v in configs.items() if k[0] == OLD]
    single = old_runs and all(v["cpu"] < 1.2 * v["wall"] for v in old_runs)
    ax.set_title("Run time by number of threads" + (" (2.0.2 uses one)" if single else ""))

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.text(-0.02, 1.035, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right",
                va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    # Headline: the numbers a reader should take away, computed from whatever has been measured.
    big = max(cells[MAIN]) if cells[MAIN] else None
    head, sub_head = f"harmonypy {NEXT} on Tahoe-100M", []
    if big is not None:
        what = f"all {big:.1f} million cells" if big == full else f"{big:.0f} million cells"
        head += f": {what} in {fmt_time(cells[MAIN][big]['wall'])}"
    if shared:
        x = shared[-1]
        where = "on all cells" if x == full else f"on {x:.0f} million cells"
        o, m = cells[OLD][x], cells[MAIN][x]
        why = ""
        if o["iterations"] != m["iterations"]:
            why = (f", where 2.0.2 needed {'/'.join(map(str, o['iterations']))} iterations and the {NEXT} "
                   f"{'/'.join(map(str, m['iterations']))}")
        sub_head.append(f"{o['wall'] / m['wall']:.0f}× faster than 2.0.2 {where} ({fmt_time(o['wall'])}{why})")
        if why:
            same = [x for x in shared if cells[OLD][x]["iterations"] == cells[MAIN][x]["iterations"]]
            if same:
                y = same[-1]
                sub_head.append(f"{cells[OLD][y]['wall'] / cells[MAIN][y]['wall']:.0f}× faster on {y:.0f} million "
                                f"cells, with the same number of iterations")
    one = dict(threads["16M"]).get(1)
    old16 = default(configs, OLD, "16M")
    if one and old16 and old16["wall"] > 1.1 * one["wall"]:
        sub_head.append(f"{old16['wall'] / one['wall']:.1f}× faster even on one thread (16M cells)")
    corr = agreement(os.path.join(os.path.dirname(os.path.abspath(a.results)), "Z"))
    if corr:
        # Round down, so the printed bound stays true (0.999996 prints as 0.9999, not 1.0000).
        bound = np.floor(min(r for r, _ in corr.values()) * 1e4) / 1e4
        sub_head.append(f"same results as 2.0.2 (per-PC correlation ≥ {bound:.4f})")
    fig.text(0.012, 0.985, head, fontsize=15, fontweight="bold", va="top")
    lines, line = [], ""
    for clause in sub_head:
        if line and len(line) + 2 + len(clause) > 120:
            lines.append(line + ";")
            line = clause
        else:
            line = f"{line}; {clause}" if line else clause
    if line:
        lines.append(line)
    if lines:
        fig.text(0.012, 0.950, "\n".join(lines), fontsize=11.5, va="top", color="#333333", linespacing=1.3)
    handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[b].items() if k not in ("label", "zorder")})
               for b in ("0.2", OLD, MAIN)]
    fig.legend(handles, [STYLE[b]["label"] for b in ("0.2", OLD, MAIN)], loc="upper left", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.005, 0.950 - 0.024 * max(1, len(lines)) - 0.004),
               handlelength=2.6, columnspacing=2.0)

    def repeats(build):
        r = sorted({v["repeats"] for k, v in configs.items() if k[0] == build})
        if not r:
            return None
        if len(r) == 1:
            return "one run" if r == [1] else f"median of {r[0]} runs"
        return f"median of {r[0]}–{r[-1]} runs" if r[0] > 1 else f"one to {r[-1]} runs, median"

    reps = "; ".join(f"{name}: {repeats(b)}" for b, name in ((MAIN, NEXT), (OLD, "2.0.2")) if repeats(b))
    builds = [name for b, name in ((OLD, "2.0.2"), (MAIN, f"the {NEXT}")) if any(k[0] == b for k in configs)]
    iters = iteration_note(cells, samples, full)
    note = (
        f"Data: Tahoe-100M, 50 PCs, corrected for sample; the 1–16 million cell subsamples have 800 samples, all "
        f"{full:.1f} million cells have {full_batches:,} (dotted segments). Default settings"
        + (f"; for {' and '.join(builds)}, {iters}" if iters else "") + ". "
        f"0.2: April 2026 benchmark on the same machine, not run on all cells; its defaults do more clustering work "
        f"(epsilon_harmony 1e-4 instead of 1e-2, up to 20 k-means rounds instead of 4), and its memory is what the "
        f"process held after the run, not its peak. Run time: run_harmony plus reading the corrected coordinates "
        f"(h.Z_corr), loading excluded ({reps}). Peak memory: the whole process, input included. Machine: a shared "
        f"server, 2× AMD EPYC 7543 (64 cores, 128 threads), 3 TB RAM. The {NEXT} (commit {COMMIT}) used all 128 "
        f"threads, its default, except in d."
    )
    fig.text(0.012, 0.085, textwrap.fill(note, 200), fontsize=8, color="#444444", va="top", linespacing=1.35)
    fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{a.out}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_summary(a.out, configs, failed, hist, cells, samples, threads, corr, full, most)


def spread(v):
    return "" if v["repeats"] == 1 else f" ({fmt_time(v['wall_min'])}–{fmt_time(v['wall_max'])}, n={v['repeats']})"


def write_summary(out, configs, failed, hist, cells, samples, threads, corr, full, most):
    hist_cells = {n / 1e6: t for n, t in hist.get("cells", [])}
    hist_samples = dict(hist.get("batches", []))
    lines = [f"# Tahoe-100M benchmark: harmonypy {NEXT} (commit {COMMIT}) vs 2.0.2", "",
             "Run time is run_harmony plus reading h.Z_corr (loading excluded): the median of the repeats, with "
             "the range and count. C++ is the time inside the compiled backend; the rest of run_harmony is "
             "Python preprocessing (mostly encoding the sample labels). Memory is the peak of the whole process.",
             "",
             "| Dataset | Cells | Samples | 0.2 (Apr 2026) | 2.0.2 | next release | 2.0.2 / next | "
             "next: C++ / Python / Z_corr | Peak memory 2.0.2 / next | Iterations |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [(d, samples, int(d[:-1]), hist_samples.get(int(d[:-1]))) for d in SAMPLE_SETS]
    rows += [(d, cells, full if d == "full" else float(d[:-1]),
              hist_cells.get(full if d == "full" else float(d[:-1]))) for d in CELL_SETS]
    for d, table, x, h in rows:
        m, o = table[MAIN].get(x), table[OLD].get(x)
        if not (m or o):
            continue
        ref = m or o
        ratio = f"{o['wall'] / m['wall']:.0f}x" if (m and o) else ""
        split = (f"{fmt_time(m['cpp'])} / {fmt_time(max(0.0, m['run'] - m['cpp']))} / {fmt_time(m['extract'])}"
                 if m else "")
        mem = " / ".join(f"{v['peak']:.1f}" if v else "–" for v in (o, m)) + " GB"
        its = ", ".join(f"{b}: {'/'.join(map(str, v['iterations']))}" for b, v in (("2.0.2", o), ("next", m)) if v)
        lines.append(
            f"| {d} | {ref['n_cells']:,} | {ref['n_batches']} | {fmt_time(h) if h else ''} | "
            f"{fmt_time(o['wall']) + spread(o) if o else ''} | {fmt_time(m['wall']) + spread(m) if m else ''} | "
            f"{ratio} | {split} | {mem} | {its} |")
    for d in ("1M", "16M"):
        if threads[d]:
            first_n, first = threads[d][0]
            lines += ["", f"## Threads, {d} cells (next release)", "",
                      f"| Threads | Run time | Speedup over {first_n} thread{'s' if first_n > 1 else ''} | "
                      "CPU time | Other users' CPU during the run |", "|---:|---:|---:|---:|---:|"]
            for n, v in threads[d]:
                lines.append(f"| {n} | {fmt_time(v['wall'])}{spread(v)} | {first['wall'] / v['wall']:.1f}x | "
                             f"{fmt_time(v['cpu'])} | {fmt_time(v['others'])} |")
    extras = [(k, v) for k, v in configs.items() if k[3] != "{}" or v["available_cpus"] != most]
    if extras:
        lines += ["", "## Other settings", "",
                  "| Build | Dataset | Settings | CPUs | Threads | Run time | Iterations | Peak memory |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for (b, d, t, kw, cpus), v in sorted(extras, key=lambda kv: (kv[0][3], kv[0][1], kv[0][0], kv[0][4])):
            lines.append(f"| {b} | {d} | {kw} | {cpus} | {t or ''} | {fmt_time(v['wall'])}{spread(v)} | "
                         f"{'/'.join(map(str, v['iterations']))} | {v['peak']:.1f} GB |")
    if corr:
        lines += ["", "## Agreement with 2.0.2", ""]
        for d, (r, rel) in corr.items():
            lines.append(f"- {d} cells: minimum per-PC correlation {r:.8f} (1 - r = {1 - r:.1e}); RMS "
                         f"difference {rel:.2e} of the coordinates' spread")
    factorize = os.path.join(os.path.dirname(os.path.abspath(out)), "results-factorize.jsonl")
    if os.path.exists(factorize):
        rs = [json.loads(line) for line in open(factorize)]
        lines += ["", "## Encoding the sample labels with pd.factorize", "",
                  "run_harmony encodes each batch variable with np.unique, which sorts the labels as strings. "
                  "venv-factorize is the same build with that line changed to pd.factorize(..., sort=True), which "
                  "gives identical codes. The two alternated, three runs each.", "",
                  "| Dataset | np.unique (as merged) | pd.factorize | Python preprocessing, np.unique / pd.factorize |",
                  "|---|---:|---:|---:|"]
        for d in ("1M", "16M", "full"):
            row = {}
            for b in ("main", "factorize"):
                x = [r for r in rs if r["dataset"] == d and r["build"] == b and not r.get("failed")]
                if x:
                    walls = [r["wall_seconds"] for r in x]
                    row[b] = (statistics.median(walls), min(walls), max(walls), len(walls),
                              statistics.median(r["run_harmony_seconds"] - r["cpp_seconds"] for r in x))
            if len(row) == 2:
                cell = lambda t: f"{fmt_time(t[0])} ({fmt_time(t[1])}–{fmt_time(t[2])}, n={t[3]})"
                lines.append(f"| {d} | {cell(row['main'])} | {cell(row['factorize'])} | "
                             f"{fmt_time(row['main'][4])} / {fmt_time(row['factorize'][4])} |")
    shas = sorted({(k[0], s) for k, v in configs.items() for s in v["sha"]})
    lines += ["", "## Builds", ""] + [f"- {b}: extension sha256 {s}…" for b, s in shas]
    if failed:
        lines += ["", "## Failed runs", ""] + [
            f"- {r.get('build', '')} {r.get('dataset', '')} {r.get('error') or r.get('command')}" for r in failed]
    text = "\n".join(lines) + "\n"
    with open(f"{out}-summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
