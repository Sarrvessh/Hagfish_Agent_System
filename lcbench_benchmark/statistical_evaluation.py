"""Statistical evaluation pipeline for multi-fidelity HPO benchmark outputs.

Expected input columns:
- algorithm
- scenario
- seed
- final_error
- anytime_auc
- cost_to_1_percent

The script computes:
1) Friedman global significance test
2) Nemenyi post-hoc analysis (average ranks + critical difference)
3) Wilcoxon pairwise tests (Hagfish vs baselines)
4) Win/Tie/Loss tallies across scenarios
5) Aggregate descriptive statistics and speedup factors
6) Markdown-ready summary tables and text verdicts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


@dataclass
class TestResult:
    """Container for pairwise hypothesis-test output."""

    metric: str
    baseline: str
    n_pairs: int
    statistic: float
    p_value: float
    significant: bool
    direction: str


class BenchmarkEvaluator:
    """Evaluate algorithm performance with non-parametric statistical tests."""

    REQUIRED_COLUMNS = {
        "algorithm",
        "scenario",
        "seed",
        "final_error",
        "anytime_auc",
        "cost_to_1_percent",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        alpha: float = 0.05,
        hagfish_name: str = "hagfish",
        penalty_multiplier: float = 1.10,
    ) -> None:
        self.alpha = alpha
        self.hagfish_name = hagfish_name
        self.penalty_multiplier = penalty_multiplier

        self.df = self._validate_and_standardize(data)
        self.df_penalized = self._penalize_missing_cost(self.df.copy())

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        alpha: float = 0.05,
        hagfish_name: str = "hagfish",
        penalty_multiplier: float = 1.10,
    ) -> "BenchmarkEvaluator":
        """Create evaluator from CSV."""
        data = pd.read_csv(csv_path)
        return cls(
            data=data,
            alpha=alpha,
            hagfish_name=hagfish_name,
            penalty_multiplier=penalty_multiplier,
        )

    def _validate_and_standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = data.copy()
        df["algorithm"] = df["algorithm"].astype(str).str.strip().str.lower()
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

        for metric in ["final_error", "anytime_auc", "cost_to_1_percent"]:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        if self.hagfish_name.lower() not in set(df["algorithm"]):
            raise ValueError(
                f"Hagfish algorithm '{self.hagfish_name}' not found in column 'algorithm'"
            )

        return df

    def _penalize_missing_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing cost_to_1_percent with scenario-level penalty.

        This handles cases where an algorithm never reaches the 1% target.
        Penalty = scenario_max_cost * penalty_multiplier.
        """

        df["cost_was_imputed"] = False

        for scenario, grp in df.groupby("scenario"):
            scenario_max = grp["cost_to_1_percent"].max(skipna=True)
            if np.isnan(scenario_max):
                # If an entire scenario has NaN costs, use a conservative fallback.
                scenario_max = 1.0

            penalty_value = scenario_max * self.penalty_multiplier
            mask = (df["scenario"] == scenario) & (df["cost_to_1_percent"].isna())
            df.loc[mask, "cost_to_1_percent"] = penalty_value
            df.loc[mask, "cost_was_imputed"] = True

        return df

    @staticmethod
    def _block_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Create block-design matrix indexed by (scenario, seed)."""

        matrix = df.pivot_table(
            index=["scenario", "seed"],
            columns="algorithm",
            values=metric,
            aggfunc="mean",
        )
        return matrix

    def run_friedman(self, metric: str) -> Dict[str, float]:
        """Global significance test across all algorithms."""

        matrix = self._block_matrix(self.df_penalized, metric=metric).dropna()
        if matrix.shape[1] < 3:
            raise ValueError("Friedman test requires at least 3 algorithms")
        if matrix.shape[0] < 2:
            raise ValueError("Friedman test requires at least 2 blocks")

        arrays = [matrix[col].values for col in matrix.columns]
        stat, p_value = stats.friedmanchisquare(*arrays)

        return {
            "metric": metric,
            "n_blocks": int(matrix.shape[0]),
            "n_algorithms": int(matrix.shape[1]),
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
        }

    def run_nemenyi(self, metric: str) -> Tuple[pd.DataFrame, float]:
        """Nemenyi post-hoc test and critical difference.

        Returns:
            - p-value matrix (algorithms x algorithms)
            - critical difference (CD)
        """

        matrix = self._block_matrix(self.df_penalized, metric=metric).dropna()
        if matrix.empty:
            raise ValueError("No complete blocks available for Nemenyi test")

        # Some scikit-posthocs versions are strict about DataFrame index names.
        # Passing ndarray avoids index-schema assumptions; we restore labels below.
        nemenyi_raw = sp.posthoc_nemenyi_friedman(matrix.to_numpy())
        nemenyi_p = pd.DataFrame(
            nemenyi_raw.values,
            index=matrix.columns,
            columns=matrix.columns,
        )

        # CD for Nemenyi: q_alpha * sqrt(k*(k+1)/(6*N)).
        # q_alpha from studentized range; division by sqrt(2) follows standard
        # conversion used for Nemenyi critical-difference diagrams.
        k = matrix.shape[1]
        n = matrix.shape[0]
        q_alpha = stats.studentized_range.ppf(1 - self.alpha, k, np.inf) / np.sqrt(2)
        cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))

        return nemenyi_p, float(cd)

    def average_ranks(self, metric: str) -> pd.Series:
        """Average rank per algorithm (lower metric is better)."""

        matrix = self._block_matrix(self.df_penalized, metric=metric).dropna()
        ranks = matrix.rank(axis=1, method="average", ascending=True)
        return ranks.mean(axis=0).sort_values()

    def wilcoxon_vs_hagfish(
        self,
        metric: str,
        baselines: Iterable[str],
    ) -> pd.DataFrame:
        """Pairwise Wilcoxon signed-rank tests for Hagfish vs baselines."""

        matrix = self._block_matrix(self.df_penalized, metric=metric)
        hagfish = self.hagfish_name.lower()
        if hagfish not in matrix.columns:
            raise ValueError(f"'{hagfish}' not found in block matrix")

        out: List[TestResult] = []

        for baseline in baselines:
            b = baseline.lower()
            if b not in matrix.columns:
                continue

            paired = matrix[[hagfish, b]].dropna()
            if paired.empty:
                continue

            x = paired[hagfish].values
            y = paired[b].values

            # Exact=False is robust for ties/zeros and moderate sample sizes.
            stat, p_value = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            diff = np.median(y - x)  # Positive means Hagfish is better for minimization
            direction = "tie"
            if diff > 0:
                direction = "hagfish_better"
            elif diff < 0:
                direction = "baseline_better"

            out.append(
                TestResult(
                    metric=metric,
                    baseline=b,
                    n_pairs=len(paired),
                    statistic=float(stat),
                    p_value=float(p_value),
                    significant=bool(p_value < self.alpha),
                    direction=direction,
                )
            )

        return pd.DataFrame([r.__dict__ for r in out])

    def win_tie_loss_vs_hagfish(
        self,
        metric: str,
        baselines: Iterable[str],
    ) -> pd.DataFrame:
        """Scenario-wise Win/Tie/Loss by Wilcoxon significance.

        For each baseline and scenario, perform Wilcoxon over seeds.
        Count outcomes across scenarios:
        - Win: Hagfish significantly better
        - Loss: baseline significantly better
        - Tie: not significant (or identical medians)
        """

        hagfish = self.hagfish_name.lower()
        rows = []

        for baseline in baselines:
            b = baseline.lower()
            wins = ties = losses = 0

            for scenario, grp in self.df_penalized.groupby("scenario"):
                pivot = grp.pivot_table(
                    index="seed",
                    columns="algorithm",
                    values=metric,
                    aggfunc="mean",
                )

                if hagfish not in pivot.columns or b not in pivot.columns:
                    continue

                paired = pivot[[hagfish, b]].dropna()
                if len(paired) < 2:
                    continue

                x = paired[hagfish].values
                y = paired[b].values
                stat, p_value = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

                if p_value < self.alpha:
                    if np.median(y - x) > 0:
                        wins += 1
                    elif np.median(y - x) < 0:
                        losses += 1
                    else:
                        ties += 1
                else:
                    ties += 1

            rows.append(
                {
                    "baseline": b,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wtl": f"{wins}/{ties}/{losses}",
                }
            )

        return pd.DataFrame(rows).sort_values("baseline").reset_index(drop=True)

    def descriptive_stats(self) -> pd.DataFrame:
        """Mean ± std for key metrics by algorithm."""

        grouped = self.df_penalized.groupby("algorithm", as_index=False).agg(
            mean_cost_to_1_percent=("cost_to_1_percent", "mean"),
            std_cost_to_1_percent=("cost_to_1_percent", "std"),
            mean_anytime_auc=("anytime_auc", "mean"),
            std_anytime_auc=("anytime_auc", "std"),
            mean_final_error=("final_error", "mean"),
            std_final_error=("final_error", "std"),
        )
        return grouped.sort_values("algorithm").reset_index(drop=True)

    def speedup_vs_hagfish(self, baselines: Iterable[str]) -> pd.DataFrame:
        """Compute speedup factors relative to Hagfish for cost_to_1_percent.

        speedup = baseline_mean_cost / hagfish_mean_cost
        (>1 means Hagfish is faster / lower cost)
        """

        stats_df = self.descriptive_stats().set_index("algorithm")
        hagfish = self.hagfish_name.lower()
        if hagfish not in stats_df.index:
            raise ValueError("Hagfish not present in descriptive stats")

        hagfish_cost = stats_df.loc[hagfish, "mean_cost_to_1_percent"]
        rows = []
        for baseline in baselines:
            b = baseline.lower()
            if b not in stats_df.index:
                continue
            baseline_cost = stats_df.loc[b, "mean_cost_to_1_percent"]
            speedup = baseline_cost / hagfish_cost
            rows.append(
                {
                    "baseline": b,
                    "hagfish_speedup_x": float(speedup),
                    "statement": (
                        f"Hagfish is {speedup:.2f}x faster than {b} "
                        f"on mean cost_to_1_percent"
                    ),
                }
            )

        return pd.DataFrame(rows)

    def markdown_summary_table(
        self,
        rank_metric: str = "final_error",
        wtl_metric: str = "cost_to_1_percent",
        baselines: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Build Markdown-ready summary table for paper drafting."""

        if baselines is None:
            baselines = ["dehb", "asha", "bohb", "random_search"]

        avg_rank = self.average_ranks(rank_metric).rename("average_rank")
        desc = self.descriptive_stats().set_index("algorithm")
        wtl = self.win_tie_loss_vs_hagfish(wtl_metric, baselines).set_index("baseline")

        rows = []
        for alg in avg_rank.index:
            mean_c = desc.loc[alg, "mean_cost_to_1_percent"]
            std_c = desc.loc[alg, "std_cost_to_1_percent"]
            cost_pm = f"{mean_c:.4f} ± {std_c:.4f}"

            if alg == self.hagfish_name.lower():
                wtl_str = "-"
            elif alg in wtl.index:
                wtl_str = wtl.loc[alg, "wtl"]
            else:
                wtl_str = "n/a"

            rows.append(
                {
                    "Algorithm": alg,
                    "Average Rank": round(float(avg_rank.loc[alg]), 4),
                    "Mean Cost to 1% ± Std": cost_pm,
                    "Win/Tie/Loss vs Hagfish": wtl_str,
                }
            )

        out = pd.DataFrame(rows).sort_values("Average Rank").reset_index(drop=True)
        return out

    def verdict_text(
        self,
        metrics: Iterable[str],
        baselines: Iterable[str],
    ) -> List[str]:
        """Generate concise textual statistical verdicts."""

        lines: List[str] = []

        for metric in metrics:
            w = self.wilcoxon_vs_hagfish(metric=metric, baselines=baselines)
            if w.empty:
                continue

            for _, row in w.iterrows():
                p = row["p_value"]
                b = row["baseline"]
                n = int(row["n_pairs"])

                if row["significant"] and row["direction"] == "hagfish_better":
                    lines.append(
                        f"For {metric}, Hagfish is significantly better than {b} "
                        f"(Wilcoxon p={p:.4g}, n={n})."
                    )
                elif row["significant"] and row["direction"] == "baseline_better":
                    lines.append(
                        f"For {metric}, {b} is significantly better than Hagfish "
                        f"(Wilcoxon p={p:.4g}, n={n})."
                    )
                else:
                    lines.append(
                        f"For {metric}, Hagfish vs {b} is not statistically significant "
                        f"(Wilcoxon p={p:.4g}, n={n})."
                    )

        return lines


def _make_dummy_data(seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic benchmark table to demonstrate end-to-end usage."""

    rng = np.random.default_rng(seed)
    algorithms = ["hagfish", "dehb", "asha", "bohb", "random_search"]
    scenarios = [f"dataset_{i}" for i in range(1, 7)]
    seeds = list(range(10))

    rows = []
    for sc in scenarios:
        scenario_shift = rng.normal(0, 0.01)
        for s in seeds:
            for alg in algorithms:
                if alg == "hagfish":
                    final_error = 0.150 + scenario_shift + rng.normal(0, 0.004)
                    anytime_auc = 35 + rng.normal(0, 2.0)
                    cost = 120 + rng.normal(0, 8.0)
                elif alg == "dehb":
                    final_error = 0.148 + scenario_shift + rng.normal(0, 0.006)
                    anytime_auc = 34 + rng.normal(0, 2.5)
                    cost = 165 + rng.normal(0, 14.0)
                elif alg == "asha":
                    final_error = 0.153 + scenario_shift + rng.normal(0, 0.006)
                    anytime_auc = 36 + rng.normal(0, 2.3)
                    cost = 145 + rng.normal(0, 10.0)
                elif alg == "bohb":
                    final_error = 0.152 + scenario_shift + rng.normal(0, 0.006)
                    anytime_auc = 36 + rng.normal(0, 2.4)
                    cost = 150 + rng.normal(0, 11.0)
                else:
                    final_error = 0.173 + scenario_shift + rng.normal(0, 0.010)
                    anytime_auc = 47 + rng.normal(0, 3.5)
                    cost = 240 + rng.normal(0, 20.0)

                # Simulate failures to reach 1% target for random_search.
                if alg == "random_search" and rng.random() < 0.35:
                    cost = np.nan

                rows.append(
                    {
                        "algorithm": alg,
                        "scenario": sc,
                        "seed": s,
                        "final_error": final_error,
                        "anytime_auc": anytime_auc,
                        "cost_to_1_percent": cost,
                    }
                )

    return pd.DataFrame(rows)


def _print_title(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


if __name__ == "__main__":
    # Demo pipeline with synthetic data.
    demo_df = _make_dummy_data(seed=42)
    evaluator = BenchmarkEvaluator(
        data=demo_df,
        alpha=0.05,
        hagfish_name="hagfish",
        penalty_multiplier=1.10,
    )

    baselines = ["dehb", "asha", "bohb"]

    _print_title("Global Significance (Friedman)")
    friedman_final = evaluator.run_friedman("final_error")
    friedman_cost = evaluator.run_friedman("cost_to_1_percent")
    print(pd.DataFrame([friedman_final, friedman_cost]).to_string(index=False))

    _print_title("Nemenyi Post-Hoc (Average Ranks + CD)")
    avg_ranks = evaluator.average_ranks("final_error")
    nemenyi_pvals, cd = evaluator.run_nemenyi("final_error")
    print("Average ranks (final_error):")
    print(avg_ranks.to_string())
    print(f"\nCritical Difference (alpha=0.05): {cd:.4f}")
    print("\nNemenyi p-value matrix:")
    print(nemenyi_pvals.to_string())

    _print_title("Wilcoxon Pairwise: Hagfish vs Baselines")
    w_final = evaluator.wilcoxon_vs_hagfish("final_error", baselines)
    w_cost = evaluator.wilcoxon_vs_hagfish("cost_to_1_percent", baselines)
    print("final_error:")
    print(w_final.to_string(index=False))
    print("\ncost_to_1_percent:")
    print(w_cost.to_string(index=False))

    _print_title("Win/Tie/Loss Matrix")
    wtl = evaluator.win_tie_loss_vs_hagfish("cost_to_1_percent", baselines)
    print(wtl.to_string(index=False))

    _print_title("Descriptive Statistics")
    desc = evaluator.descriptive_stats()
    print(desc.to_string(index=False))

    _print_title("Speedup Factors")
    speedup = evaluator.speedup_vs_hagfish(["dehb", "asha"])
    print(speedup.to_string(index=False))

    _print_title("Markdown Summary Table")
    md_table = evaluator.markdown_summary_table(
        rank_metric="final_error",
        wtl_metric="cost_to_1_percent",
        baselines=["dehb", "asha", "bohb", "random_search"],
    )
    try:
        print(md_table.to_markdown(index=False))
    except Exception:
        print(md_table.to_string(index=False))

    _print_title("Textual Statistical Verdicts")
    for line in evaluator.verdict_text(
        metrics=["final_error", "cost_to_1_percent"],
        baselines=baselines,
    ):
        print("-", line)
