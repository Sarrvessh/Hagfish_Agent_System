"""Orchestrate Optuna + Ray + Hagfish runs into one merged standardized CSV.

This script runs the three existing entrypoints sequentially:
- run_optuna_baselines.py
- run_ray_baselines.py
- run_hagfish.py

Each stage writes to a temporary standardized CSV, and the orchestrator appends
those rows into a single merged standardized CSV with schema:

    [dataset, algorithm, seed, cumulative_simulated_cost, best_validation_error]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

STAGE_COLUMNS = [
    "algorithm",
    "seed",
    "cumulative_simulated_cost",
    "best_validation_error",
]

MERGED_COLUMNS = ["dataset", *STAGE_COLUMNS]


def _run_subprocess(command: Sequence[str]) -> None:
    """Run one stage command."""

    print("\n[orchestrator] Running:", " ".join(command))
    subprocess.run(command, check=True)


def _validate_standard_csv(path: Path) -> pd.DataFrame:
    """Load and validate standardized CSV schema."""

    if not path.exists():
        raise FileNotFoundError(f"Stage output CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in STAGE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Stage CSV {path} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df[STAGE_COLUMNS].copy()


def _append_standard_csv(source: Path, merged: Path, dataset_name: str) -> int:
    """Append standardized rows from source into merged CSV."""

    df = _validate_standard_csv(source)
    df.insert(0, "dataset", str(dataset_name))
    df = df[MERGED_COLUMNS]
    merged.parent.mkdir(parents=True, exist_ok=True)

    write_header = not merged.exists()
    df.to_csv(merged, mode="a", header=write_header, index=False)
    return len(df)


def _run_stage_and_append(
    stage_name: str,
    command: Sequence[str],
    stage_output_csv: Path,
    merged_csv: Path,
    dataset_name: str,
) -> dict:
    """Run one stage and append its output if available.

    Returns a summary dictionary for final reporting.
    """

    report = {
        "stage": stage_name,
        "status": "failed",
        "rows_appended": 0,
        "error": "",
    }

    try:
        _run_subprocess(command)
    except subprocess.CalledProcessError as exc:
        report["error"] = f"runner failed with exit code {exc.returncode}"
        return report

    try:
        rows = _append_standard_csv(stage_output_csv, merged_csv, dataset_name)
        report["status"] = "ok"
        report["rows_appended"] = int(rows)
        return report
    except Exception as exc:  # noqa: BLE001 - summary should include any append error
        report["error"] = f"append failed: {exc}"
        return report


def _stage_optuna_command(
    dataset_name: str,
    benchmark_seed: int | None,
    seeds: Sequence[int],
    fidelity_name: str,
    min_budget: float,
    max_budget: float,
    num_fidelity_steps: int,
    metric_key: str | None,
    cost_key: str | None,
    output_csv: Path,
    algorithms: Sequence[str],
    n_trials: int,
) -> List[str]:
    """Build CLI command for the Optuna baseline stage."""

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_optuna_baselines.py")),
        "--dataset-name",
        dataset_name,
        "--algorithms",
        *list(algorithms),
        "--seeds",
        *[str(s) for s in seeds],
        "--n-trials",
        str(n_trials),
        "--fidelity-name",
        fidelity_name,
        "--min-budget",
        str(min_budget),
        "--max-budget",
        str(max_budget),
        "--num-fidelity-steps",
        str(num_fidelity_steps),
        "--output-csv",
        str(output_csv),
    ]

    if benchmark_seed is not None:
        cmd.extend(["--benchmark-seed", str(benchmark_seed)])

    if metric_key is not None:
        cmd.extend(["--metric-key", metric_key])
    if cost_key is not None:
        cmd.extend(["--cost-key", cost_key])

    return cmd


def _stage_ray_command(
    dataset_name: str,
    benchmark_seed: int | None,
    seeds: Sequence[int],
    fidelity_name: str,
    min_budget: float,
    max_budget: float,
    num_fidelity_steps: int,
    metric_key: str | None,
    cost_key: str | None,
    output_csv: Path,
    algorithms: Sequence[str],
    n_samples: int,
    pbt_population_size: int,
    epsilon: float,
) -> List[str]:
    """Build CLI command for the Ray baseline stage."""

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_ray_baselines.py")),
        "--dataset-name",
        dataset_name,
        "--algorithms",
        *list(algorithms),
        "--seeds",
        *[str(s) for s in seeds],
        "--n-samples",
        str(n_samples),
        "--pbt-population-size",
        str(pbt_population_size),
        "--epsilon",
        str(epsilon),
        "--fidelity-name",
        fidelity_name,
        "--min-budget",
        str(min_budget),
        "--max-budget",
        str(max_budget),
        "--num-fidelity-steps",
        str(num_fidelity_steps),
        "--output-csv",
        str(output_csv),
    ]

    if benchmark_seed is not None:
        cmd.extend(["--benchmark-seed", str(benchmark_seed)])

    if metric_key is not None:
        cmd.extend(["--metric-key", metric_key])
    if cost_key is not None:
        cmd.extend(["--cost-key", cost_key])

    return cmd


def _stage_hagfish_command(
    dataset_name: str,
    benchmark_seed: int | None,
    seeds: Sequence[int],
    fidelity_name: str,
    min_budget: float,
    max_budget: float,
    num_fidelity_steps: int,
    metric_key: str | None,
    cost_key: str | None,
    output_csv: Path,
    n_trials: int,
    epsilon: float,
) -> List[str]:
    """Build CLI command for the Hagfish stage."""

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_hagfish.py")),
        "--dataset-name",
        dataset_name,
        "--seeds",
        *[str(s) for s in seeds],
        "--n-trials",
        str(n_trials),
        "--epsilon",
        str(epsilon),
        "--fidelity-name",
        fidelity_name,
        "--min-budget",
        str(min_budget),
        "--max-budget",
        str(max_budget),
        "--num-fidelity-steps",
        str(num_fidelity_steps),
        "--output-csv",
        str(output_csv),
    ]

    if benchmark_seed is not None:
        cmd.extend(["--benchmark-seed", str(benchmark_seed)])

    if metric_key is not None:
        cmd.extend(["--metric-key", metric_key])
    if cost_key is not None:
        cmd.extend(["--cost-key", cost_key])

    return cmd


def parse_args() -> argparse.Namespace:
    """Parse orchestrator CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna + Ray + Hagfish sequentially and append all standardized "
            "outputs into one merged CSV."
        )
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
        "--stages",
        nargs="+",
        default=["optuna", "ray", "hagfish"],
        choices=["optuna", "ray", "hagfish"],
        help="Subset of stages to run sequentially.",
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--fidelity-name", type=str, default="budget")
    parser.add_argument("--min-budget", type=float, default=0.1)
    parser.add_argument("--max-budget", type=float, default=1.0)
    parser.add_argument("--num-fidelity-steps", type=int, default=5)
    parser.add_argument("--metric-key", type=str, default=None)
    parser.add_argument("--cost-key", type=str, default=None)

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(REPOSITORY_ROOT / "results" / "hpobench" / "results_merged.csv"),
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=str(REPOSITORY_ROOT / "tmp" / "hpobench"),
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to existing merged CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep stage temporary CSV files.",
    )

    parser.add_argument(
        "--optuna-algorithms",
        nargs="+",
        default=["sha", "hyperband", "tpe"],
        choices=["sha", "hyperband", "tpe"],
    )
    parser.add_argument("--optuna-n-trials", type=int, default=30)

    parser.add_argument(
        "--ray-algorithms",
        nargs="+",
        default=["pbt", "epsilon_greedy"],
        choices=["pbt", "epsilon_greedy"],
    )
    parser.add_argument("--ray-n-samples", type=int, default=20)
    parser.add_argument("--ray-pbt-population-size", type=int, default=8)
    parser.add_argument("--ray-epsilon", type=float, default=0.2)

    parser.add_argument("--hagfish-n-trials", type=int, default=30)
    parser.add_argument("--hagfish-epsilon", type=float, default=0.2)

    return parser.parse_args()


def main() -> None:
    """Entrypoint for sequential orchestration and merged CSV appending."""

    args = parse_args()

    merged_path = Path(args.output_csv)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if merged_path.exists() and not args.append_existing:
        merged_path.unlink()

    appended_rows = 0
    stage_outputs: List[Path] = []
    stage_reports: List[dict] = []

    if "optuna" in args.stages:
        optuna_out = tmp_dir / "optuna_results.csv"
        stage_outputs.append(optuna_out)
        optuna_cmd = _stage_optuna_command(
            dataset_name=args.dataset_name,
            benchmark_seed=args.benchmark_seed,
            seeds=args.seeds,
            fidelity_name=args.fidelity_name,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            num_fidelity_steps=args.num_fidelity_steps,
            metric_key=args.metric_key,
            cost_key=args.cost_key,
            output_csv=optuna_out,
            algorithms=args.optuna_algorithms,
            n_trials=args.optuna_n_trials,
        )
        report = _run_stage_and_append(
            stage_name="optuna",
            command=optuna_cmd,
            stage_output_csv=optuna_out,
            merged_csv=merged_path,
            dataset_name=args.dataset_name,
        )
        stage_reports.append(report)
        appended_rows += int(report["rows_appended"])

    if "ray" in args.stages:
        ray_out = tmp_dir / "ray_results.csv"
        stage_outputs.append(ray_out)
        ray_cmd = _stage_ray_command(
            dataset_name=args.dataset_name,
            benchmark_seed=args.benchmark_seed,
            seeds=args.seeds,
            fidelity_name=args.fidelity_name,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            num_fidelity_steps=args.num_fidelity_steps,
            metric_key=args.metric_key,
            cost_key=args.cost_key,
            output_csv=ray_out,
            algorithms=args.ray_algorithms,
            n_samples=args.ray_n_samples,
            pbt_population_size=args.ray_pbt_population_size,
            epsilon=args.ray_epsilon,
        )
        report = _run_stage_and_append(
            stage_name="ray",
            command=ray_cmd,
            stage_output_csv=ray_out,
            merged_csv=merged_path,
            dataset_name=args.dataset_name,
        )
        stage_reports.append(report)
        appended_rows += int(report["rows_appended"])

    if "hagfish" in args.stages:
        hagfish_out = tmp_dir / "hagfish_results.csv"
        stage_outputs.append(hagfish_out)
        hagfish_cmd = _stage_hagfish_command(
            dataset_name=args.dataset_name,
            benchmark_seed=args.benchmark_seed,
            seeds=args.seeds,
            fidelity_name=args.fidelity_name,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            num_fidelity_steps=args.num_fidelity_steps,
            metric_key=args.metric_key,
            cost_key=args.cost_key,
            output_csv=hagfish_out,
            n_trials=args.hagfish_n_trials,
            epsilon=args.hagfish_epsilon,
        )
        report = _run_stage_and_append(
            stage_name="hagfish",
            command=hagfish_cmd,
            stage_output_csv=hagfish_out,
            merged_csv=merged_path,
            dataset_name=args.dataset_name,
        )
        stage_reports.append(report)
        appended_rows += int(report["rows_appended"])

    if not args.keep_temp:
        for path in stage_outputs:
            if path.exists():
                path.unlink()

    print(
        "\n[orchestrator] Completed sequential run. "
        f"Appended {appended_rows} rows into merged CSV: {merged_path}"
    )

    print("[orchestrator] Stage summary:")
    for report in stage_reports:
        if report["status"] == "ok":
            print(
                f"  - {report['stage']}: OK "
                f"(rows_appended={report['rows_appended']})"
            )
        else:
            print(
                f"  - {report['stage']}: FAILED "
                f"({report['error']})"
            )

    failed = [r for r in stage_reports if r["status"] != "ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
