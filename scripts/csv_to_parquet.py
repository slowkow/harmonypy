#!/usr/bin/env python
"""Convert CSV metadata files in data/tahoe-ilya/ to Parquet format.

Usage:
    python scripts/csv_to_parquet.py
"""

import glob
import os
from time import time

import pandas as pd


def convert_csv_to_parquet(csv_path):
    parquet_path = csv_path.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        print(f"  Skipping (already exists): {parquet_path}")
        return

    csv_size = os.path.getsize(csv_path)
    print(f"  Reading: {csv_path} ({csv_size / 1e6:.0f} MB)")

    start = time()
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path)
    elapsed = time() - start

    parquet_size = os.path.getsize(parquet_path)
    ratio = csv_size / parquet_size
    print(f"  Wrote:   {parquet_path} ({parquet_size / 1e6:.0f} MB, {ratio:.1f}x smaller, {elapsed:.1f}s)")


def main():
    base = "data/tahoe-ilya"
    csv_files = sorted(glob.glob(f"{base}/**/*.csv", recursive=True))

    if not csv_files:
        print(f"No CSV files found in {base}/")
        return

    print(f"Found {len(csv_files)} CSV files\n")
    for csv_path in csv_files:
        convert_csv_to_parquet(csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
