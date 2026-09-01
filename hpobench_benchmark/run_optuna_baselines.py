"""Run native Optuna multi-fidelity baselines on simple-hpo-bench datasets.

Baselines implemented with official Optuna APIs:
- TPE + SuccessiveHalvingPruner
- TPE + HyperbandPruner
- TPE + NopPruner

Each run writes standardized anytime trajectories via logger.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import optuna

from env_bridge import HPOBenchObjectiveFunction
from logger import UnifiedResultLogger


def build_pruner(name: str, num_fidelity_steps: int) -> optuna.pruners.BasePruner:
    """Create native Optuna pruner from CLI label."""

    lname = name.lower()
    if lname == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    if lname == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, int(num_fidelity_steps)),
            reduction_factor=3,
        )
    if lname == "tpe":
        return optuna.pruners.NopPruner()

    raise ValueError(f"Unknown Optuna algorithm: {name}")


def run_one_optuna_algorithm(
    algorithm: str,
    seed: int,
    n_trials: int,
    bridge: HPOBenchObjectiveFunction,
    results_logger: UnifiedResultLogger,
) -> None:
    """Execute one Optuna baseline for a single seed."""

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = build_pruner(
        name=algorithm,
        num_fidelity_steps=bridge.num_fidelity_steps,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{algorithm}_seed_{seed}",
    )

    schedule = bridge.fidelity_schedule()

    def objective(trial: optuna.trial.Trial) -> float:
        config = bridge.optuna_sample_from_trial(trial)
        best_error = float("inf")

        for step_idx, budget in enumerate(schedule):
            result = bridge.evaluate(config=config, fidelity=budget, seed=seed)
            best_error = min(best_error, result.validation_error)

            # Unified anytime logging across all engines.
            results_logger.log_observation(
                algorithm=algorithm,
                seed=seed,
                incremental_simulated_cost=result.simulated_cost,
                validation_error=best_error,
            )

            # Native Optuna pruning signal at each fidelity step.
            trial.report(best_error, step=step_idx + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_error

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run native Optuna baselines on simple-hpo-bench datasets."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="simple-hpo-bench dataset name, e.g. 'australian'",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=None,
        help="Optional seed passed to hpo_benchmarks.HPOBench.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["sha", "hyperband", "tpe"],
        choices=["sha", "hyperband", "tpe"],
        help="Optuna baseline variants.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--fidelity-name", type=str, default="budget")
    parser.add_argument("--min-budget", type=float, default=0.1)
    parser.add_argument("--max-budget", type=float, default=1.0)
    parser.add_argument("--num-fidelity-steps", type=int, default=5)
    parser.add_argument("--metric-key", type=str, default=None)
    parser.add_argument("--cost-key", type=str, default=None)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiments/output/simple_hpo_bench_native/results.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for Optuna baseline execution."""

    args = parse_args()
    bridge = HPOBenchObjectiveFunction(
        dataset_name=args.dataset_name,
        benchmark_seed=args.benchmark_seed,
        fidelity_name=args.fidelity_name,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        num_fidelity_steps=args.num_fidelity_steps,
        metric_key=args.metric_key,
        cost_key=args.cost_key,
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_logger = UnifiedResultLogger(output_csv=args.output_csv)

    for algorithm in args.algorithms:
        for seed in args.seeds:
            # Reset stream keeps each algorithm/seed independent in cumulative cost.
            results_logger.reset_stream(algorithm=algorithm, seed=seed)
            run_one_optuna_algorithm(
                algorithm=algorithm,
                seed=seed,
                n_trials=args.n_trials,
                bridge=bridge,
                results_logger=results_logger,
            )

    print(f"Saved standardized results to: {results_logger.output_csv}")


if __name__ == "__main__":
    main()
