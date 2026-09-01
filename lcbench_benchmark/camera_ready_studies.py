"""HAT ablation and alpha-sensitivity study on YAHPO LCBench.

The study isolates the planner--critic--memory controller. Hyperparameter
configurations are sampled from a paired deterministic stream; HAT controls the
maximum fidelity queried for each trial. This keeps the configuration proposal
mechanism fixed while testing the budget policy itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault(
    "YAHPO_LOCAL_CONFIG",
    str(Path(__file__).resolve().parent / "yahpo_local_config.yml"),
)

from adaptive_trainer import AdaptiveTrainer
from benchmark_runner import YAHPOScenario, geometric_fidelity_levels


DEFAULT_INSTANCES = ["3945", "167104", "168329", "168908", "189873"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_ALPHAS = [0.0, 1e-4, 1e-3, 1e-2]

ABLATIONS: Dict[str, Dict[str, bool]] = {
    "Full HAT": {},
    "No memory": {"use_memory": False},
    "No elite attraction": {"use_elite": False},
    "No burst exploration": {"use_burst": False},
    "No cost awareness": {"use_cost_control": False},
}


def holm_adjust(p_values: Iterable[float]) -> List[float]:
    values = np.nan_to_num(np.asarray(list(p_values), dtype=float), nan=1.0)
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
    denom = float(ranks.sum())
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / denom)


def plan_to_fidelity(plan: Dict[str, int], minimum: float, maximum: float) -> float:
    """Map the documented iteration budget [50, 1000] to benchmark fidelity."""
    normalized = np.clip((float(plan["max_iter"]) - 50.0) / 950.0, 0.0, 1.0)
    target = float(minimum) + normalized * (float(maximum) - float(minimum))
    return float(np.clip(round(target), minimum, maximum))


def run_controller(
    instance: str,
    seed: int,
    trials: int,
    alpha: float,
    variant: str,
    flags: Dict[str, bool],
) -> Dict[str, float | int | str]:
    scenario = YAHPOScenario("lcbench", instance, seed)
    rng = np.random.default_rng(seed)
    trainer = AdaptiveTrainer(alpha=alpha, seed=seed, **flags)

    best_error = float("inf")
    total_cost = 0.0
    evaluations = 0
    max_fidelities: List[float] = []
    pop_budgets: List[int] = []
    burst_events = 0
    started = time.perf_counter()

    problem_size = max(1, len(scenario.opt_space.get_hyperparameters()))
    for trial_id in range(trials):
        was_stagnated = trainer.memory.stagnation_count >= 3
        plan = trainer.plan({"problem_size": problem_size, "dataset_size": problem_size})
        if was_stagnated and flags.get("use_burst", True):
            burst_events += 1

        target = plan_to_fidelity(plan, scenario.min_fidelity, scenario.max_fidelity)
        levels = geometric_fidelity_levels(scenario.min_fidelity, target)
        config = scenario.sample_random_config(rng)
        trial_error = float("inf")
        trial_cost = 0.0

        for fidelity in levels:
            observation = scenario.evaluate(config, fidelity)
            trial_error = min(trial_error, observation.validation_error)
            trial_cost += observation.cost
            evaluations += 1

        best_error = min(best_error, trial_error)
        total_cost += trial_cost
        # LCBench exposes validation accuracy in percentage units; the adapter
        # stores 1-accuracy, so negating and scaling gives a higher-is-better score.
        metric = -best_error / 100.0
        trainer.observe(
            metric=metric,
            cost=trial_cost,
            params=plan,
            episode=trial_id + 1,
        )
        max_fidelities.append(target)
        pop_budgets.append(int(plan["pop_size"]))

    elapsed = time.perf_counter() - started
    return {
        "study": "ablation" if variant in ABLATIONS else "sensitivity",
        "variant": variant,
        "instance": str(instance),
        "seed": int(seed),
        "alpha": float(alpha),
        "trials": int(trials),
        "final_best_error": float(best_error),
        "total_cost": float(total_cost),
        "runtime_seconds": float(elapsed),
        "objective_evaluations": int(evaluations),
        "mean_max_fidelity": float(np.mean(max_fidelities)),
        "mean_population_budget": float(np.mean(pop_budgets)),
        "burst_events": int(burst_events),
    }


def summarize(df: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = (
        df.groupby(group, sort=False)
        .agg(
            mean_final_best_error=("final_best_error", "mean"),
            std_final_best_error=("final_best_error", "std"),
            mean_total_cost=("total_cost", "mean"),
            std_total_cost=("total_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_objective_evaluations=("objective_evaluations", "mean"),
            mean_max_fidelity=("mean_max_fidelity", "mean"),
            mean_burst_events=("burst_events", "mean"),
            n_runs=("seed", "size"),
        )
        .reset_index()
    )
    return summary


def ablation_tests(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["variant"] == "Full HAT"].sort_values(["instance", "seed"])
    rows = []
    for variant in ABLATIONS:
        if variant == "Full HAT":
            continue
        other = df[df["variant"] == variant].sort_values(["instance", "seed"])
        for metric in ["final_best_error", "total_cost"]:
            x = full[metric].to_numpy()
            y = other[metric].to_numpy()
            if np.allclose(x, y, rtol=0.0, atol=0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "comparison": f"Full HAT vs {variant}",
                    "metric": metric,
                    "n_pairs": int(len(x)),
                    "statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_full_minus_variant": rank_biserial(x, y),
                }
            )
    out = pd.DataFrame(rows)
    out["holm_p_value"] = 1.0
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "holm_p_value"] = holm_adjust(out.loc[idx, "p_value"])
    out["significant_0_05"] = out["holm_p_value"] < 0.05
    return out


def write_latex_tables(ablation: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> None:
    ablation_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Final transformed objective$\downarrow$ & Cost$\downarrow$ & Time (s)$\downarrow$ & $n$ \\",
        r"\midrule",
    ]
    for _, row in ablation.iterrows():
        ablation_lines.append(
            f"{row['variant']} & {row['mean_final_best_error']:.3f} & "
            f"{row['mean_total_cost']:.0f} & {row['mean_runtime_seconds']:.3f} & "
            f"{int(row['n_runs'])} \\\\" 
        )
    ablation_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "ablation_table.tex").write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")

    sensitivity_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\alpha$ & Final transformed objective$\downarrow$ & Cost$\downarrow$ & Time (s)$\downarrow$ & $n$ \\",
        r"\midrule",
    ]
    for _, row in sensitivity.iterrows():
        sensitivity_lines.append(
            f"{row['alpha']:.0e} & {row['mean_final_best_error']:.3f} & "
            f"{row['mean_total_cost']:.0f} & {row['mean_runtime_seconds']:.3f} & "
            f"{int(row['n_runs'])} \\\\" 
        )
    sensitivity_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "alpha_sensitivity_table.tex").write_text("\n".join(sensitivity_lines) + "\n", encoding="utf-8")


def make_figure(ablation: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))

    axes[0].scatter(
        ablation["mean_total_cost"],
        ablation["mean_final_best_error"],
        color="black",
        s=28,
    )
    for _, row in ablation.iterrows():
        axes[0].annotate(
            row["variant"].replace(" attraction", "").replace(" exploration", ""),
            (row["mean_total_cost"], row["mean_final_best_error"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[0].set_xlabel("Mean simulated cost (lower is better)")
    axes[0].set_ylabel("Mean final best error (lower is better)")
    axes[0].set_title("(a) Controller ablations")
    axes[0].grid(alpha=0.25)

    alpha_labels = [f"{value:.0e}" if value else "0" for value in sensitivity["alpha"]]
    x = np.arange(len(sensitivity))
    axes[1].plot(x, sensitivity["mean_final_best_error"], "o-", color="black", label="Final transformed objective")
    cost_axis = axes[1].twinx()
    cost_axis.plot(x, sensitivity["mean_total_cost"], "s--", color="0.45", label="Cost")
    axes[1].set_xticks(x, alpha_labels)
    axes[1].set_xlabel(r"Cost sensitivity $\alpha$")
    axes[1].set_ylabel("Final best error")
    cost_axis.set_ylabel("Simulated cost")
    axes[1].set_title("(b) Alpha sensitivity")
    axes[1].grid(alpha=0.25)
    lines = axes[1].lines + cost_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(output / "ablation_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "ablation_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "camera_ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | int | str]] = []

    for variant, flags in ABLATIONS.items():
        for instance in args.instances:
            for seed in args.seeds:
                print(f"[ablation] {variant} instance={instance} seed={seed}", flush=True)
                rows.append(run_controller(instance, seed, args.trials, args.alpha, variant, flags))

    for alpha in args.alphas:
        label = f"alpha={alpha:g}"
        for instance in args.instances:
            for seed in args.seeds:
                print(f"[sensitivity] {label} instance={instance} seed={seed}", flush=True)
                row = run_controller(instance, seed, args.trials, alpha, label, {})
                row["study"] = "sensitivity"
                rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "controller_study_runs.csv", index=False)
    ablation_raw = raw[raw["study"] == "ablation"].copy()
    sensitivity_raw = raw[raw["study"] == "sensitivity"].copy()
    ablation_summary = summarize(ablation_raw, "variant")
    sensitivity_summary = summarize(sensitivity_raw, "alpha").sort_values("alpha")
    tests = ablation_tests(ablation_raw)

    ablation_summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    sensitivity_summary.to_csv(args.output_dir / "alpha_sensitivity_summary.csv", index=False)
    tests.to_csv(args.output_dir / "ablation_wilcoxon_holm.csv", index=False)
    write_latex_tables(ablation_summary, sensitivity_summary, args.output_dir)
    make_figure(ablation_summary, sensitivity_summary, args.output_dir)

    metadata = {
        "protocol": "Paired deterministic configuration streams; HAT controls maximum LCBench fidelity per trial.",
        "instances": list(map(str, args.instances)),
        "instance_selection": "Existing-results stratification spanning low/high cost and easier/harder instances.",
        "seeds": args.seeds,
        "trials": args.trials,
        "ablation_alpha": args.alpha,
        "sensitivity_alphas": args.alphas,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (args.output_dir / "study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote camera-ready study outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
