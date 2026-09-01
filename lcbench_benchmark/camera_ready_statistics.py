"""Paired non-parametric statistics for the saved 34-instance LCBench runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values: Iterable[float]) -> List[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.abs(diff) > 0]
    if not len(diff):
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def load_final_runs(path: Path) -> pd.DataFrame:
    state: Dict[Tuple[str, str, int], Dict[str, float | int | str]] = {}
    usecols = ["algorithm", "instance", "seed", "best_validation_error", "cumulative_cost"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        grouped = chunk.groupby(["algorithm", "instance", "seed"], sort=False)
        for key, group in grouped:
            current = state.get(key)
            best_error = float(group["best_validation_error"].min())
            total_cost = float(group["cumulative_cost"].max())
            if current is None:
                state[key] = {
                    "algorithm": str(key[0]),
                    "instance": str(key[1]),
                    "seed": int(key[2]),
                    "final_best_error": best_error,
                    "total_cost": total_cost,
                }
            else:
                current["final_best_error"] = min(float(current["final_best_error"]), best_error)
                current["total_cost"] = max(float(current["total_cost"]), total_cost)
    return pd.DataFrame(state.values())


def load_final_table(path: Path) -> pd.DataFrame:
    required = {
        "algorithm",
        "instance",
        "seed",
        "final_best_error",
        "total_cost",
    }
    runs = pd.read_csv(path)
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    runs = runs[list(required)].copy()
    runs["algorithm"] = (
        runs["algorithm"].astype(str).str.lower().replace({"hagfish": "hat"})
    )
    runs["instance"] = runs["instance"].astype(str)
    runs["seed"] = runs["seed"].astype(int)
    return runs


def paired_tests(runs: pd.DataFrame, metric: str, baselines: Iterable[str]) -> pd.DataFrame:
    matrix = runs.pivot(index=["instance", "seed"], columns="algorithm", values=metric)
    rows = []
    for baseline in baselines:
        paired = matrix[["hat", baseline]].dropna()
        x = paired["hat"].to_numpy()
        y = paired[baseline].to_numpy()
        try:
            result = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
        median_difference = float(np.median(x - y))
        rows.append(
            {
                "metric": metric,
                "comparison": f"HAT vs {baseline.upper()}",
                "n_pairs": int(len(paired)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "median_hat_minus_baseline": median_difference,
                "rank_biserial_hat_minus_baseline": rank_biserial(x, y),
            }
        )
    out = pd.DataFrame(rows)
    out["holm_p_value"] = holm_adjust(out["p_value"])
    out["significant_0_05"] = out["holm_p_value"] < 0.05
    return out


def friedman_and_ranks(
    runs: pd.DataFrame, metric: str, algorithms: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(algorithms)
    matrix = runs.pivot(
        index=["instance", "seed"], columns="algorithm", values=metric
    )[columns].dropna()
    result = stats.friedmanchisquare(*[matrix[col].to_numpy() for col in matrix.columns])
    global_test = pd.DataFrame(
        [{
            "metric": metric,
            "n_blocks": int(matrix.shape[0]),
            "n_algorithms": int(matrix.shape[1]),
            "friedman_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }]
    )
    ranks = (
        matrix.rank(axis=1, ascending=True, method="average")
        .mean(axis=0)
        .sort_values()
        .rename("average_rank")
        .reset_index()
    )
    ranks.insert(0, "metric", metric)
    return global_test, ranks


def write_report(pairwise: pd.DataFrame, global_tests: pd.DataFrame, ranks: pd.DataFrame, output: Path) -> None:
    lines = [
        "# LCBench Statistical Analysis",
        "",
        "Paired unit: LCBench instance and seed (34 instances x 10 seeds = 340 blocks).",
        "Two-sided Wilcoxon signed-rank tests compare HAT with the archived DEHB-style baseline and corrected ASHA; Holm correction is applied separately per metric.",
        "Rank-biserial signs use HAT minus baseline; positive values mean higher, worse HAT values because both metrics are minimized.",
        "",
        "## Pairwise tests",
        "",
        pairwise.to_markdown(index=False),
        "",
        "## Friedman tests",
        "",
        global_tests.to_markdown(index=False),
        "",
        "## Average ranks",
        "",
        ranks.to_markdown(index=False),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "lcbench"
        / "final_runs.csv",
        help="Saved per-run final metrics; this command never launches experiments.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "camera_ready")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_final_table(args.input_csv)
    baselines = ["dehb_style", "asha"]
    expected = {"hat", "dehb_style", "asha", "random_search"}
    if set(runs["algorithm"]) != expected:
        raise RuntimeError(
            f"Expected algorithms {sorted(expected)}, found {sorted(set(runs['algorithm']))}"
        )
    if len(runs) != 1360 or runs.groupby("algorithm").size().ne(340).any():
        raise RuntimeError("Expected 340 paired runs for each of four algorithms")
    pairwise = pd.concat(
        [paired_tests(runs, metric, baselines) for metric in ["final_best_error", "total_cost"]],
        ignore_index=True,
    )
    global_parts, rank_parts = [], []
    for metric in ["final_best_error", "total_cost"]:
        global_test, ranks = friedman_and_ranks(
            runs, metric, ["hat", "dehb_style", "asha"]
        )
        global_parts.append(global_test)
        rank_parts.append(ranks)
    global_tests = pd.concat(global_parts, ignore_index=True)
    ranks = pd.concat(rank_parts, ignore_index=True)

    pairwise.to_csv(args.output_dir / "lcbench_pairwise_wilcoxon_holm.csv", index=False)
    global_tests.to_csv(args.output_dir / "lcbench_friedman.csv", index=False)
    ranks.to_csv(args.output_dir / "lcbench_average_ranks.csv", index=False)
    write_report(pairwise, global_tests, ranks, args.output_dir / "STATISTICAL_REPORT.md")
    print(pairwise.to_string(index=False))
    print(global_tests.to_string(index=False))


if __name__ == "__main__":
    main()
