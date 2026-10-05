# Tahoe-100M benchmark: harmonypy 2.1.0 (commit 623ff51) vs 2.0.2

Run time is run_harmony plus reading h.Z_corr (loading excluded): the median of the repeats, with the range and count. C++ is the time inside the compiled backend; the rest of run_harmony is Python preprocessing (mostly encoding the sample labels). Memory is the peak of the whole process.

| Dataset | Cells | Samples | 0.2 (Apr 2026) | 2.0.2 | 2.1.0 | 2.0.2 / 2.1.0 | 2.1.0: C++ / Python / Z_corr | Peak memory 2.0.2 / 2.1.0 | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50B | 1,000,000 | 50 | 5.0 min | 58 s | 4.1 s (3.9 s–4.2 s, n=5) | 14x | 3.0 s / 0.7 s / 0.3 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 100B | 1,000,000 | 100 | 4.3 min | 58 s | 4.1 s (3.9 s–4.1 s, n=5) | 14x | 3.0 s / 0.7 s / 0.4 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 200B | 1,000,000 | 200 | 6.4 min | 59 s | 4.0 s (3.8 s–4.0 s, n=5) | 15x | 2.8 s / 0.8 s / 0.3 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 400B | 1,000,000 | 400 | 11 min | 60 s | 3.9 s (3.8 s–4.0 s, n=5) | 15x | 2.8 s / 0.7 s / 0.3 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 800B | 1,000,000 | 800 | 23 min | 1.0 min | 4.0 s (3.8 s–4.1 s, n=5) | 15x | 3.0 s / 0.7 s / 0.3 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 1M | 1,000,000 | 800 | 21 min | 1.0 min (1.0 min–1.0 min, n=3) | 3.9 s (3.7 s–4.1 s, n=5) | 16x | 2.9 s / 0.7 s / 0.4 s | 3.3 / 2.4 GB | 2.0.2: 1, 2.1.0: 1 |
| 2M | 2,000,000 | 800 | 35 min | 2.1 min | 6.7 s (6.6 s–6.8 s, n=5) | 19x | 4.6 s / 1.5 s / 0.6 s | 6.4 / 4.7 GB | 2.0.2: 1, 2.1.0: 1 |
| 4M | 4,000,000 | 800 | 55 min | 4.3 min | 12 s (12 s–12 s, n=5) | 21x | 7.9 s / 3.1 s / 1.2 s | 12.7 / 9.3 GB | 2.0.2: 1, 2.1.0: 1 |
| 8M | 8,000,000 | 800 | 1.7 h | 8.7 min | 23 s (23 s–23 s, n=5) | 23x | 14 s / 6.6 s / 2.3 s | 25.2 / 18.1 GB | 2.0.2: 1, 2.1.0: 1 |
| 16M | 16,000,000 | 800 | 3.8 h | 18 min (18 min–19 min, n=3) | 40 s (40 s–41 s, n=5) | 28x | 21 s / 14 s / 4.4 s | 50.0 / 36.0 GB | 2.0.2: 1, 2.1.0: 1 |
| full | 95,596,109 | 1344 |  | 4.5 h | 4.9 min (3.9 min–6.4 min, n=3) | 55x | 2.0 min / 1.6 min / 31 s | 294.5 / 214.3 GB | 2.0.2: 3, 2.1.0: 1 |

## Threads, 1M cells (2.1.0)

| Threads | Run time | Speedup over 1 thread | CPU time | Other users' CPU during the run |
|---:|---:|---:|---:|---:|
| 1 | 20 s (20 s–20 s, n=3) | 1.0x | 20 s | 2.9 min |
| 2 | 11 s (11 s–11 s, n=3) | 1.8x | 21 s | 1.6 min |
| 4 | 7.2 s (6.3 s–7.3 s, n=3) | 2.7x | 25 s | 52 s |
| 8 | 4.6 s (4.4 s–5.1 s, n=3) | 4.3x | 26 s | 41 s |
| 16 | 3.6 s (3.4 s–3.7 s, n=3) | 5.4x | 32 s | 24 s |
| 32 | 3.4 s (3.3 s–3.7 s, n=3) | 5.7x | 49 s | 22 s |
| 64 | 3.7 s (3.5 s–3.7 s, n=3) | 5.3x | 1.4 min | 22 s |
| 128 | 3.9 s (3.7 s–4.1 s, n=5) | 5.1x | 2.4 min | 0.3 s |

## Threads, 16M cells (2.1.0)

| Threads | Run time | Speedup over 1 thread | CPU time | Other users' CPU during the run |
|---:|---:|---:|---:|---:|
| 1 | 5.5 min (5.4 min–6.0 min, n=3) | 1.0x | 5.5 min | 36 min |
| 2 | 3.1 min (3.0 min–3.1 min, n=3) | 1.8x | 5.9 min | 18 min |
| 4 | 1.8 min (1.7 min–1.9 min, n=3) | 3.1x | 5.9 min | 12 min |
| 8 | 1.1 min (1.1 min–1.1 min, n=3) | 5.0x | 5.9 min | 9.9 min |
| 16 | 50 s (48 s–53 s, n=3) | 6.7x | 6.6 min | 7.7 min |
| 32 | 42 s (41 s–50 s, n=3) | 7.9x | 7.8 min | 6.6 min |
| 64 | 39 s (39 s–40 s, n=3) | 8.4x | 12 min | 6.4 min |
| 128 | 40 s (40 s–41 s, n=5) | 8.3x | 20 min | 18 s |

## Other settings

| Build | Dataset | Settings | CPUs | Threads | Run time | Iterations | Peak memory |
|---|---|---|---|---:|---:|---:|---:|
| main-623ff51 | 16M | {"epsilon_cluster": 1e-05, "epsilon_harmony": 0.0001, "lamb": 1, "max_iter_kmeans": 20} | 0-127 | 128 | 1.4 min (1.4 min–1.4 min, n=3) | 10 | 36.1 GB |
| main-623ff51 | 1M | {"epsilon_cluster": 1e-05, "epsilon_harmony": 0.0001, "lamb": 1, "max_iter_kmeans": 20} | 0-127 | 128 | 7.4 s (7.4 s–7.7 s, n=3) | 6 | 2.4 GB |
| main-623ff51 | 16M | {"epsilon_harmony": -1, "max_iter_harmony": 10} | 0-127 | 128 | 1.3 min (1.3 min–1.3 min, n=3) | 10 | 36.1 GB |
| 2.0.2 | 1M | {"epsilon_harmony": -1, "max_iter_harmony": 10} | 0-127 |  | 5.1 min | 10 | 3.3 GB |
| main-623ff51 | 1M | {"epsilon_harmony": -1, "max_iter_harmony": 10} | 0-127 | 128 | 8.7 s (8.7 s–9.7 s, n=3) | 10 | 2.4 GB |
| main-623ff51 | 16M | {} | 0-31 | 32 | 40 s (40 s–40 s, n=3) | 1 | 36.1 GB |
| main-623ff51 | 16M | {} | 0-31,64-95 | 64 | 40 s (40 s–40 s, n=3) | 1 | 36.0 GB |
| main-623ff51 | 16M | {} | 0-63 | 64 | 39 s (39 s–40 s, n=3) | 1 | 36.0 GB |

## Agreement with 2.0.2

- 1M cells: minimum per-PC correlation 1.00000000 (1 - r = 5.3e-11); RMS difference 6.96e-06 of the coordinates' spread
- 16M cells: minimum per-PC correlation 1.00000000 (1 - r = 4.6e-10); RMS difference 2.10e-05 of the coordinates' spread

## Encoding the sample labels with pd.factorize

run_harmony encodes each batch variable with np.unique, which sorts the labels as strings. venv-factorize is the same build with that line changed to pd.factorize(..., sort=True), which gives identical codes. The two alternated, three runs each.

| Dataset | np.unique (as merged) | pd.factorize | Python preprocessing, np.unique / pd.factorize |
|---|---:|---:|---:|
| 1M | 4.0 s (4.0 s–4.1 s, n=3) | 3.3 s (3.3 s–3.5 s, n=3) | 0.7 s / 0.2 s |
| 16M | 40 s (40 s–41 s, n=3) | 28 s (28 s–29 s, n=3) | 14 s / 2.4 s |
| full | 3.8 min (3.8 min–3.9 min, n=3) | 2.7 min (2.6 min–4.8 min, n=3) | 1.5 min / 17 s |

## Builds

- 2.0.2: extension sha256 013ee054ff1f90ce…
- main-623ff51: extension sha256 e226215cbcaa2cb4…
