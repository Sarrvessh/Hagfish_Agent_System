"""Create a publication-ready anytime performance plot from benchmark logs.

Expected input columns:
- algorithm
- seed
- cumulative_simulated_cost
- best_validation_error

The script aligns asynchronous trajectories by interpolating step functions onto
a common, dense, log-spaced cost grid before computing mean/std across seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ALGO_STYLE: Dict[str, Dict[str, object]] = {
    "hagfish": {"color": "#000000", "marker": "X", "linewidth": 2.5},
    "hyperband": {"color": "#d62728", "marker": "s", "linewidth": 2.0},
    "sha": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0},
    "tpe": {"color": "#2ca02c", "marker": "^", "linewidth": 2.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready anytime performance plot"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to results_merged.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="publication_plot.png",
        help="Output PNG path (default: publication_plot.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Anytime Performance (HPOBench)",
        help="Figure title",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=500,
        help="Number of log-spaced cost grid points",
    )
    return parser.parse_args()


def build_common_log_grid(cost: np.ndarray, n_grid: int) -> np.ndarray:
    """Build a dense, common, log-spaced cost grid across all runs."""

    positive = cost[np.isfinite(cost) & (cost > 0)]
    if positive.size == 0:
        raise ValueError("No positive cost values were found in the CSV.")

    c_min = float(np.min(positive))
    c_max = float(np.max(positive))
    if c_min == c_max:
        c_max = c_min * 1.0001

    return np.logspace(np.log10(c_min), np.log10(c_max), int(max(50, n_grid)))


def step_interpolate_forward_fill(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a best-so-far step function onto a shared cost grid."""

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Enforce best-so-far monotonicity and remove duplicate x by keeping best value.
    y_sorted = np.minimum.accumulate(y_sorted)

    dedup_x: List[float] = []
    dedup_y: List[float] = []
    for xi, yi in zip(x_sorted, y_sorted):
        if dedup_x and np.isclose(xi, dedup_x[-1]):
            dedup_y[-1] = min(dedup_y[-1], float(yi))
        else:
            dedup_x.append(float(xi))
            dedup_y.append(float(yi))

    x_arr = np.asarray(dedup_x, dtype=float)
    y_arr = np.asarray(dedup_y, dtype=float)

    idx = np.searchsorted(x_arr, grid, side="right") - 1
    out = np.empty_like(grid, dtype=float)

    valid = idx >= 0
    out[valid] = y_arr[idx[valid]]
    out[~valid] = y_arr[0]
    return out


def marker_positions(n_points: int, n_markers: int = 15) -> List[int]:
    """Pick visually balanced marker indices on a log-like spacing."""

    if n_points <= 1:
        return [0]
    idx = np.geomspace(1, n_points, num=min(n_markers, n_points)) - 1
    uniq = np.unique(np.clip(np.round(idx).astype(int), 0, n_points - 1))
    return uniq.tolist()


def aggregate_on_common_grid(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Compute mean/std trajectories per algorithm on the shared cost grid."""

    rows: List[Dict[str, float]] = []

    for algorithm, algo_df in df.groupby("algorithm", sort=True):
        per_seed_curves: List[np.ndarray] = []

        for _seed, run_df in algo_df.groupby("seed", sort=True):
            x = run_df["cumulative_simulated_cost"].to_numpy(dtype=float)
            y = run_df["best_validation_error"].to_numpy(dtype=float)

            keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
            x = x[keep]
            y = y[keep]
            if x.size == 0:
                continue

            curve = step_interpolate_forward_fill(x=x, y=y, grid=grid)
            per_seed_curves.append(curve)

        if not per_seed_curves:
            continue

        stack = np.vstack(per_seed_curves)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)

        for cost_i, mean_i, std_i in zip(grid, mean, std):
            rows.append(
                {
                    "algorithm": str(algorithm),
                    "cumulative_simulated_cost": float(cost_i),
                    "mean_best_validation_error": float(mean_i),
                    "std_best_validation_error": float(std_i),
                }
            )

    return pd.DataFrame(rows)


def plot_anytime(agg: pd.DataFrame, title: str, output_path: Path) -> None:
    """Render and save the publication-quality anytime figure."""

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=300)

    present_algorithms = sorted(agg["algorithm"].unique())

    for algorithm in present_algorithms:
        sub = agg[agg["algorithm"] == algorithm].sort_values("cumulative_simulated_cost")

        x = sub["cumulative_simulated_cost"].to_numpy(dtype=float)
        mean = sub["mean_best_validation_error"].to_numpy(dtype=float)
        std = sub["std_best_validation_error"].to_numpy(dtype=float)

        style = ALGO_STYLE.get(
            algorithm,
            {"color": "#7f7f7f", "marker": "o", "linewidth": 1.8},
        )

        ax.plot(
            x,
            mean,
            label=algorithm,
            color=style["color"],
            linewidth=float(style["linewidth"]),
            marker=str(style["marker"]),
            markevery=marker_positions(len(x), n_markers=15),
            markersize=5.8,
            zorder=4 if algorithm == "hagfish" else 3,
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative Simulated Cost", fontsize=13)
    ax.set_ylabel("Best Validation Error", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="best", frameon=True, fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.csv_path)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {
        "algorithm",
        "seed",
        "cumulative_simulated_cost",
        "best_validation_error",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only declared columns and expected algorithms for clean plotting.
    df = df[[
        "algorithm",
        "seed",
        "cumulative_simulated_cost",
        "best_validation_error",
    ]].copy()

    target_algorithms = ["hagfish", "hyperband", "sha", "tpe"]
    df = df[df["algorithm"].isin(target_algorithms)]
    if df.empty:
        raise ValueError("No rows left after filtering to expected algorithms.")

    common_grid = build_common_log_grid(
        df["cumulative_simulated_cost"].to_numpy(dtype=float),
        n_grid=args.n_grid,
    )
    agg = aggregate_on_common_grid(df=df, grid=common_grid)

    output_path = Path(args.output)
    plot_anytime(agg=agg, title=args.title, output_path=output_path)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
