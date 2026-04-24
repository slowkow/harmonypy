#!/usr/bin/env python
"""
Compare harmonized outputs visually.

Generates two sets of figures:
1. UMAP embeddings colored by batch (one panel per method)
2. PC scatter plots (PC1 vs PC2, PC3 vs PC4, PC5 vs PC6) colored by batch

Usage:
    python scripts/plot_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "figures"
UMAP_SUBSAMPLE = 50_000  # subsample for UMAP (858k is too large)
SCATTER_SUBSAMPLE = 50_000  # subsample for PC scatter plots
DPI = 200
POINT_SIZE = 1.0
POINT_ALPHA = 1.0
RANDOM_STATE = 42

DATASETS = {
    "Original PCs": f"{DATA_DIR}/acute_myeloid_pcs.tsv.gz",
    "R harmony 1.2.4": f"{DATA_DIR}/acute_myeloid_pcs_harmonized.tsv.gz",
    "R harmony2": f"{DATA_DIR}/acute_myeloid_pcs_harmony2.tsv.gz",
    "harmonypy": None,  # will be computed
}


def load_pcs(path):
    """Load PC matrix, keeping only PC columns."""
    df = pd.read_csv(path, sep="\t")
    # Keep only columns that look like PCs (or all numeric if no PC columns)
    pc_cols = [c for c in df.columns if c.upper().startswith("PC")]
    if pc_cols:
        return df[pc_cols].values
    return df.select_dtypes(include=[np.number]).values


def run_harmonypy(pcs, meta):
    """Run harmonypy on the input PCs."""
    import harmonypy as hm
    print("  Running harmonypy...", flush=True)
    t0 = time()
    ho = hm.run_harmony(pcs, meta, ["batch"])
    print(f"  Done in {time() - t0:.1f}s")
    return ho.Z_corr


def compute_umap(X, n_neighbors=15, min_dist=0.5, n_epochs=2000, random_state=42):
    """Compute UMAP embedding."""
    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
        sys.exit(1)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        random_state=random_state,
        n_jobs=1,
        verbose=True,
    )
    return reducer.fit_transform(X)


def make_batch_colors(batch_labels):
    """Assign a color to each batch. Returns (color_array, palette, unique_batches).

    Uses HSV hue wheel with full saturation so 120 batches are all vivid
    and distinguishable at small point sizes.
    """
    unique = np.unique(batch_labels)
    n = len(unique)
    # Evenly spaced hues around the color wheel, full saturation/value
    hues = np.linspace(0, 1, n, endpoint=False)
    # Shuffle hues so adjacent batches (alphabetically) aren't similar colors
    rng = np.random.RandomState(42)
    rng.shuffle(hues)
    palette = plt.cm.hsv(hues)
    palette[:, 3] = 1.0  # full alpha in palette; per-point alpha set in scatter
    batch_to_idx = {b: i for i, b in enumerate(unique)}
    color_idx = np.array([batch_to_idx[b] for b in batch_labels])
    return color_idx, palette, unique


def subsample(arrays, batch_labels, n, rng):
    """Subsample arrays and batch_labels together."""
    N = len(batch_labels)
    if N <= n:
        return arrays, batch_labels
    idx = rng.choice(N, n, replace=False)
    idx.sort()
    return [a[idx] for a in arrays], batch_labels[idx]


def plot_umap_grid(embeddings, batch_labels, title_map, palette, out_path):
    """Plot a 1×4 grid of UMAP embeddings colored by batch."""
    n_panels = len(embeddings)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), dpi=DPI)
    if n_panels == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings.items()):
        color_idx, _, _ = make_batch_colors(batch_labels)
        colors = palette[color_idx]
        # Shuffle point order so no batch is always on top
        order = np.random.RandomState(0).permutation(len(emb))
        ax.scatter(
            emb[order, 0], emb[order, 1],
            c=colors[order],
            s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0, rasterized=True,
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"UMAP of acute myeloid data ({len(batch_labels):,} cells, {len(np.unique(batch_labels))} batches)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_pc_scatter_grid(pc_data_dict, batch_labels, palette, out_path):
    """Plot PC scatter grid: rows = methods, columns = PC pairs."""
    pc_pairs = [(0, 1), (2, 3), (4, 5)]
    pc_names = [("PC1", "PC2"), ("PC3", "PC4"), ("PC5", "PC6")]
    methods = list(pc_data_dict.keys())
    n_rows = len(methods)
    n_cols = len(pc_pairs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), dpi=DPI)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    color_idx, _, _ = make_batch_colors(batch_labels)
    colors = palette[color_idx]
    order = np.random.RandomState(0).permutation(len(batch_labels))

    for row, method in enumerate(methods):
        pcs = pc_data_dict[method]
        for col, ((i, j), (xname, yname)) in enumerate(zip(pc_pairs, pc_names)):
            ax = axes[row, col]
            ax.scatter(
                pcs[order, i], pcs[order, j],
                c=colors[order],
                s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0, rasterized=True,
            )
            if row == 0:
                ax.set_title(f"{xname} vs {yname}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(method, fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel(yname if row == n_rows - 1 else "")
            ax.set_xlabel(xname if row == n_rows - 1 else "")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"PC scatter plots ({len(batch_labels):,} cells, {len(np.unique(batch_labels))} batches)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_pc_correlation(pc_data, out_path):
    """Plot harmonypy PC vs other method PC for PC1–PC6.

    Rows = PC1..PC6, Columns = Original PCs, R harmony 1.2.4, R harmony2.
    All points plotted (no subsampling). Pearson r shown in each panel.
    """
    from scipy.stats import pearsonr

    compare_methods = ["Original PCs", "R harmony 1.2.4", "R harmony2"]
    n_pcs = 6
    n_rows = n_pcs
    n_cols = len(compare_methods)
    harmonypy_pcs = pc_data["harmonypy"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), dpi=DPI)

    for col, method in enumerate(compare_methods):
        other_pcs = pc_data[method]
        for row in range(n_pcs):
            ax = axes[row, col]
            x = other_pcs[:, row]
            y = harmonypy_pcs[:, row]
            r, _ = pearsonr(x, y)

            ax.scatter(
                x, y,
                s=0.05, alpha=0.3, color="black", linewidths=0, rasterized=True,
            )

            # Identity line
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color="red", linewidth=0.8, linestyle="--", alpha=0.7)

            ax.text(
                0.05, 0.95, f"r = {r:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(method, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"harmonypy PC{row + 1}", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel(f"{method}\nPC{row + 1}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)

    N = harmonypy_pcs.shape[0]
    fig.suptitle(
        f"harmonypy vs other methods — per-PC correlation ({N:,} cells)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    # Load metadata
    print("Loading metadata...")
    meta = pd.read_csv(f"{DATA_DIR}/acute_myeloid_obs.tsv.gz", sep="\t")
    batch_labels = meta["batch"].values
    N = len(batch_labels)
    print(f"  {N:,} cells, {len(np.unique(batch_labels))} batches")

    # Load all PC matrices
    print("Loading PC matrices...")
    pc_data = {}
    for name, path in DATASETS.items():
        if path is not None:
            print(f"  {name}: {path}")
            pc_data[name] = load_pcs(path)
            print(f"    shape: {pc_data[name].shape}")

    # Run harmonypy
    print("Running harmonypy on original PCs...")
    pc_data["harmonypy"] = run_harmonypy(
        pc_data["Original PCs"].copy(), meta
    )
    print(f"    shape: {pc_data['harmonypy'].shape}")

    # Build color palette once
    _, palette, _ = make_batch_colors(batch_labels)

    # --- PC correlation plots (all cells) ---
    print(f"\nGenerating PC correlation plots (all {N:,} cells)...")
    plot_pc_correlation(pc_data, f"{OUT_DIR}/pc_correlation.png")

    # --- PC scatter plots (subsampled) ---
    print(f"\nGenerating PC scatter plots (subsample={SCATTER_SUBSAMPLE:,})...")
    sub_pcs_list, sub_batch = subsample(
        [pc_data[k] for k in pc_data], batch_labels, SCATTER_SUBSAMPLE, rng
    )
    sub_pc_data = {k: v for k, v in zip(pc_data.keys(), sub_pcs_list)}
    plot_pc_scatter_grid(sub_pc_data, sub_batch, palette, f"{OUT_DIR}/pc_scatter_comparison.png")

    # --- UMAP embeddings (subsampled, cached) ---
    umap_cache = f"{OUT_DIR}/.umap_cache.npz"
    print(f"\nComputing UMAP embeddings (subsample={UMAP_SUBSAMPLE:,})...")

    # Reuse the same subsample indices for UMAP
    rng_umap = np.random.RandomState(RANDOM_STATE)
    N_total = len(batch_labels)
    if N_total > UMAP_SUBSAMPLE:
        umap_idx = rng_umap.choice(N_total, UMAP_SUBSAMPLE, replace=False)
        umap_idx.sort()
    else:
        umap_idx = np.arange(N_total)
    sub_batch_umap = batch_labels[umap_idx]
    sub_pc_dict = {k: v[umap_idx] for k, v in pc_data.items()}

    # Try loading cached UMAP embeddings
    method_names = list(sub_pc_dict.keys())
    if os.path.exists(umap_cache):
        print(f"  Loading cached UMAP from {umap_cache}")
        cached = np.load(umap_cache)
        umap_embeddings = {name: cached[name] for name in method_names if name in cached}
        if set(umap_embeddings.keys()) == set(method_names):
            print(f"  Cache hit — all {len(method_names)} embeddings loaded")
        else:
            missing = set(method_names) - set(umap_embeddings.keys())
            print(f"  Partial cache — computing {len(missing)} missing: {missing}")
            for name in missing:
                print(f"  UMAP for {name}...")
                t0 = time()
                umap_embeddings[name] = compute_umap(sub_pc_dict[name])
                print(f"    Done in {time() - t0:.1f}s")
            np.savez(umap_cache, **umap_embeddings)
    else:
        umap_embeddings = {}
        for name, pcs in sub_pc_dict.items():
            print(f"  UMAP for {name}...")
            t0 = time()
            umap_embeddings[name] = compute_umap(pcs)
            print(f"    Done in {time() - t0:.1f}s")
        np.savez(umap_cache, **umap_embeddings)
        print(f"  Saved cache to {umap_cache}")

    plot_umap_grid(umap_embeddings, sub_batch_umap, DATASETS, palette,
                   f"{OUT_DIR}/umap_comparison.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
