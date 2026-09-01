"""Comprehensive benchmark analysis with statistical inference and rankings.

This script analyzes anytime benchmark logs and produces:
- Per-run metrics (final error, log-cost AUC)
- Per-dataset algorithm summaries
- Per-dataset rankings
- Global rankings across datasets
- Friedman omnibus test across algorithms
- Pairwise Wilcoxon signed-rank tests with Holm correction
- A markdown report and CSV outputs

Input requirements:
- algorithm
- seed
- cumulative_simulated_cost
- best_validation_error

Dataset handling:
- Preferred: include a `dataset` column in input CSV.
- Alternative: pass multiple CSVs with matching --dataset-names.
- Alternative: pass one CSV with --single-dataset-name to tag all rows.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


REQUIRED_COLUMNS = {
    "algorithm",
    "seed",
    "cumulative_simulated_cost",
    "best_validation_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark logs with statistical inference and rankings"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Single CSV input path",
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=None,
        help="Multiple CSV input paths (one per dataset)",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="Dataset names matching --input-csvs order",
    )
    parser.add_argument(
        "--single-dataset-name",
        type=str,
        default=None,
        help="Dataset label to apply when input has no dataset column",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/statistics",
        help="Directory to write analysis outputs",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=400,
        help="Grid size for anytime interpolation",
    )
    parser.add_argument(
        "--ranking-metric",
        type=str,
        choices=["auc", "final"],
        default="auc",
        help="Metric used for dataset/global ranking",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, path_hint: str) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path_hint}: {missing}")


def _load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv and args.input_csvs:
        raise ValueError("Use either --input-csv or --input-csvs, not both")

    if not args.input_csv and not args.input_csvs:
        raise ValueError("Provide --input-csv or --input-csvs")

    if args.input_csv:
        path = Path(args.input_csv)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        df = pd.read_csv(path)
        _validate_columns(df, str(path))

        if "dataset" not in df.columns:
            if not args.single_dataset_name:
                raise ValueError(
                    "Input CSV has no 'dataset' column. Provide --single-dataset-name "
                    "or use --input-csvs with --dataset-names."
                )
            df = df.copy()
            df["dataset"] = args.single_dataset_name
        return df

    csv_paths = [Path(p) for p in args.input_csvs]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")

    if args.dataset_names is None:
        raise ValueError("When using --input-csvs, you must provide --dataset-names")
    if len(args.dataset_names) != len(csv_paths):
        raise ValueError("--dataset-names length must match --input-csvs length")

    frames: List[pd.DataFrame] = []
    for path, dataset in zip(csv_paths, args.dataset_names):
        df = pd.read_csv(path)
        _validate_columns(df, str(path))
        df = df.copy()
        df["dataset"] = dataset
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)


def _step_interpolate(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    y_sorted = np.minimum.accumulate(y_sorted)

    # Deduplicate same-cost points using best (minimum) error.
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


def _dataset_grid(cost: np.ndarray, n_grid: int) -> np.ndarray:
    positive = cost[np.isfinite(cost) & (cost > 0)]
    if positive.size == 0:
        raise ValueError("No positive cumulative cost values found")

    c_min = float(np.min(positive))
    c_max = float(np.max(positive))
    if c_min == c_max:
        c_max = c_min * 1.0001

    return np.logspace(np.log10(c_min), np.log10(c_max), num=max(100, int(n_grid)))


def _run_level_metrics(df: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    for dataset, ds_df in df.groupby("dataset", sort=True):
        grid = _dataset_grid(ds_df["cumulative_simulated_cost"].to_numpy(dtype=float), n_grid)

        for (algorithm, seed), run_df in ds_df.groupby(["algorithm", "seed"], sort=True):
            x = run_df["cumulative_simulated_cost"].to_numpy(dtype=float)
            y = run_df["best_validation_error"].to_numpy(dtype=float)

            keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
            x = x[keep]
            y = y[keep]
            if x.size == 0:
                continue

            curve = _step_interpolate(x=x, y=y, grid=grid)
            # Normalize AUC by span in log-cost space for comparability within dataset.
            log_grid = np.log10(grid)
            auc = np.trapz(curve, x=log_grid) / (log_grid[-1] - log_grid[0])

            final_error = float(np.min(y))

            rows.append(
                {
                    "dataset": str(dataset),
                    "algorithm": str(algorithm),
                    "seed": int(seed),
                    "final_error": float(final_error),
                    "auc_log_cost": float(auc),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid runs found after filtering")
    return out


def _dataset_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = run_metrics.groupby(["dataset", "algorithm"], sort=True)
    summary = grouped.agg(
        n_runs=("seed", "nunique"),
        mean_final_error=("final_error", "mean"),
        std_final_error=("final_error", "std"),
        mean_auc_log_cost=("auc_log_cost", "mean"),
        std_auc_log_cost=("auc_log_cost", "std"),
    ).reset_index()
    return summary


def _dataset_rankings(summary: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    metric_col = "mean_auc_log_cost" if ranking_metric == "auc" else "mean_final_error"
    rows: List[pd.DataFrame] = []
    for dataset, ds in summary.groupby("dataset", sort=True):
        ds = ds.copy()
        ds["rank"] = ds[metric_col].rank(method="average", ascending=True)
        ds = ds.sort_values("rank")
        rows.append(ds[["dataset", "algorithm", metric_col, "rank"]])
    return pd.concat(rows, ignore_index=True)


def _global_rankings(dataset_rankings: pd.DataFrame) -> pd.DataFrame:
    out = (
        dataset_rankings.groupby("algorithm", sort=True)["rank"]
        .mean()
        .reset_index(name="avg_rank")
        .sort_values("avg_rank")
        .reset_index(drop=True)
    )
    out["global_position"] = np.arange(1, len(out) + 1)
    return out[["global_position", "algorithm", "avg_rank"]]


def _friedman_test(summary: pd.DataFrame, ranking_metric: str) -> Tuple[float, float, pd.DataFrame]:
    metric_col = "mean_auc_log_cost" if ranking_metric == "auc" else "mean_final_error"
    pivot = summary.pivot_table(
        index="dataset",
        columns="algorithm",
        values=metric_col,
        aggfunc="mean",
    )
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float("nan"), float("nan"), pivot

    arrays = [pivot[col].to_numpy() for col in pivot.columns]
    stat, pvalue = friedmanchisquare(*arrays)
    return float(stat), float(pvalue), pivot


def _holm_correction(pvalues: List[float]) -> List[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for k, idx in enumerate(order):
        factor = m - k
        val = pvalues[idx] * factor
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)

    return adjusted.tolist()


def _pairwise_wilcoxon(summary: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    metric_col = "mean_auc_log_cost" if ranking_metric == "auc" else "mean_final_error"

    pivot = summary.pivot_table(
        index="dataset",
        columns="algorithm",
        values=metric_col,
        aggfunc="mean",
    ).dropna(axis=0, how="any")

    algos = list(pivot.columns)
    records: List[Dict[str, float | str]] = []

    for a, b in itertools.combinations(algos, 2):
        x = pivot[a].to_numpy(dtype=float)
        y = pivot[b].to_numpy(dtype=float)

        # Wilcoxon requires differences not all zero.
        if np.allclose(x, y):
            stat = 0.0
            pval = 1.0
        else:
            stat, pval = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
            stat = float(stat)
            pval = float(pval)

        records.append(
            {
                "algorithm_a": a,
                "algorithm_b": b,
                "wilcoxon_stat": stat,
                "p_value": pval,
            }
        )

    if not records:
        return pd.DataFrame(columns=[
            "algorithm_a",
            "algorithm_b",
            "wilcoxon_stat",
            "p_value",
            "p_value_holm",
            "significant_0_05",
        ])

    pvals = [float(r["p_value"]) for r in records]
    pvals_holm = _holm_correction(pvals)

    for rec, p_adj in zip(records, pvals_holm):
        rec["p_value_holm"] = float(p_adj)
        rec["significant_0_05"] = bool(p_adj < 0.05)

    return pd.DataFrame(records).sort_values("p_value_holm")


def _write_markdown_report(
    out_path: Path,
    ranking_metric: str,
    friedman_stat: float,
    friedman_p: float,
    global_ranks: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Statistical Report")
    lines.append("")
    lines.append(f"Ranking metric: `{ranking_metric}` (lower is better)")
    lines.append("")
    if np.isnan(friedman_stat) or np.isnan(friedman_p):
        lines.append("Friedman test: insufficient complete dataset blocks.")
    else:
        lines.append(f"Friedman chi-square: `{friedman_stat:.6f}`")
        lines.append(f"Friedman p-value: `{friedman_p:.6g}`")
        lines.append(
            "Interpretation: "
            + ("significant differences detected." if friedman_p < 0.05 else "no significant omnibus difference.")
        )
    lines.append("")
    lines.append("## Global Ranking")
    lines.append("")
    lines.append(global_ranks.to_markdown(index=False))
    lines.append("")
    lines.append("## Pairwise Wilcoxon (Holm corrected)")
    lines.append("")
    if pairwise.empty:
        lines.append("No pairwise comparisons available.")
    else:
        lines.append(pairwise.to_markdown(index=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_inputs(args)
    df = df.copy()
    df["algorithm"] = df["algorithm"].astype(str).str.lower().str.strip()

    run_metrics = _run_level_metrics(df=df, n_grid=args.n_grid)
    summary = _dataset_summary(run_metrics)
    rankings = _dataset_rankings(summary, ranking_metric=args.ranking_metric)
    global_ranks = _global_rankings(rankings)

    friedman_stat, friedman_p, _pivot = _friedman_test(summary, ranking_metric=args.ranking_metric)
    pairwise = _pairwise_wilcoxon(summary, ranking_metric=args.ranking_metric)

    run_metrics.to_csv(output_dir / "run_level_metrics.csv", index=False)
    summary.to_csv(output_dir / "dataset_algorithm_summary.csv", index=False)
    rankings.to_csv(output_dir / "dataset_rankings.csv", index=False)
    global_ranks.to_csv(output_dir / "global_rankings.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_wilcoxon_holm.csv", index=False)

    _write_markdown_report(
        out_path=output_dir / "statistical_report.md",
        ranking_metric=args.ranking_metric,
        friedman_stat=friedman_stat,
        friedman_p=friedman_p,
        global_ranks=global_ranks,
        pairwise=pairwise,
    )

    print(f"Saved analysis artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
