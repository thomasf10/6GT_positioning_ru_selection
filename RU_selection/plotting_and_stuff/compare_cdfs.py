"""Compare NMSE CDFs extracted from cdf_nmse_fixed_beams.tex.

For each pair (A, B) reports:
- Percentile gaps at 10/50/90%: NMSE_B(p) - NMSE_A(p) in dB.
  Positive => A achieves *lower* NMSE at that percentile (A is better).
- Wasserstein-1 distance: area between the CDFs, in dB.
  Signed version = (mean NMSE under B) - (mean NMSE under A); positive => A better.
- Kolmogorov-Smirnov statistic: max |F_A(x) - F_B(x)| in probability units.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).with_name("extracted_data")

CURVES = {
    "baseline_2":      "Baseline_2_Optimal_RU_fixed_beams_0_0.csv",
    "baseline_4":      "Baseline_4_Closest_RU_fixed_beams_0_0.csv",
    "approach_1_ghz":  "Approach_1_ghz_CSI_input.csv",
    "approach_1_thz":  "Approach_1_thz_CSI_input.csv",
}

PAIRS = [
    ("approach_1_ghz", "baseline_2"),
    ("approach_1_ghz", "baseline_4"),
    ("approach_1_thz", "baseline_2"),
    ("approach_1_thz", "baseline_4"),
    ("approach_1_ghz", "approach_1_thz"),
]

PERCENTILES = (0.10, 0.50, 0.90)
GRID_POINTS = 20000


def load_cdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]
    y = np.maximum.accumulate(y[order])  # guarantee monotone non-decreasing
    return x, y


def quantile(x: np.ndarray, cdf: np.ndarray, p: float) -> float:
    """Inverse CDF via linear interpolation in the CDF-value axis."""
    return float(np.interp(p, cdf, x))


def cdf_at(x_query: np.ndarray, x: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    return np.interp(x_query, x, cdf, left=0.0, right=1.0)


def shared_grid(xA, xB, n=GRID_POINTS) -> np.ndarray:
    return np.linspace(min(xA[0], xB[0]), max(xA[-1], xB[-1]), n)


def wasserstein_1_signed(xA, cdfA, xB, cdfB) -> tuple[float, float]:
    """Return (|W1|, signed W1) where signed = int(F_A - F_B) dx = E[X_B] - E[X_A].

    Signed > 0 means A has a lower mean NMSE than B, i.e. A is better.
    """
    xg = shared_grid(xA, xB)
    FA = cdf_at(xg, xA, cdfA)
    FB = cdf_at(xg, xB, cdfB)
    signed = float(np.trapz(FA - FB, xg))
    abs_w1 = float(np.trapz(np.abs(FA - FB), xg))
    return abs_w1, signed


def ks_statistic(xA, cdfA, xB, cdfB) -> float:
    xg = shared_grid(xA, xB)
    FA = cdf_at(xg, xA, cdfA)
    FB = cdf_at(xg, xB, cdfB)
    return float(np.max(np.abs(FA - FB)))


def main() -> None:
    curves = {name: load_cdf(DATA_DIR / fname) for name, fname in CURVES.items()}

    col_pair = 36
    pct_cols = "".join(f"  d@{int(p*100):02d}%[dB]" for p in PERCENTILES)
    header = f"{'pair (A vs B)':<{col_pair}}{pct_cols}  |W1|[dB]  W1s[dB]    KS"
    print(header)
    print("-" * len(header))

    for a, b in PAIRS:
        xA, cdfA = curves[a]
        xB, cdfB = curves[b]

        gaps = [quantile(xB, cdfB, p) - quantile(xA, cdfA, p) for p in PERCENTILES]
        w1_abs, w1_signed = wasserstein_1_signed(xA, cdfA, xB, cdfB)
        ks = ks_statistic(xA, cdfA, xB, cdfB)

        row = f"{a+' vs '+b:<{col_pair}}"
        for g in gaps:
            row += f"  {g:+8.2f}"
        row += f"  {w1_abs:7.2f}  {w1_signed:+7.2f}  {ks:.3f}"
        print(row)

    print()
    print("Legend:")
    print("  d@p%   = NMSE_B(p) - NMSE_A(p), gap in dB at the p-th percentile.")
    print("           Positive => A reaches percentile p at lower NMSE (A better).")
    print("  |W1|   = int |F_A - F_B| dx, unsigned area between the CDFs, in dB.")
    print("  W1s    = int (F_A - F_B) dx = E[X_B] - E[X_A], signed mean-NMSE gap in dB.")
    print("           Positive => A has lower mean NMSE (A better).")
    print("  KS     = sup_x |F_A(x) - F_B(x)|, probability units.")


if __name__ == "__main__":
    main()
