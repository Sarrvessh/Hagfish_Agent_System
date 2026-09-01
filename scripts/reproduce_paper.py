"""Safe entry point for reproducing paper analyses from saved artifacts.

The default command only displays help. Expensive experiments are never
started implicitly; ``commands`` prints the exact launch commands instead.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_script(path: Path, *arguments: str) -> None:
    subprocess.run([PYTHON, str(path), *arguments], cwd=ROOT, check=True)


def verify() -> None:
    required = [
        ROOT / "adaptive_trainer" / "optimizer.py",
        ROOT / "adaptive_trainer" / "planner.py",
        ROOT / "adaptive_trainer" / "critic.py",
        ROOT / "adaptive_trainer" / "memory.py",
        ROOT / "results" / "lcbench" / "full_hat" / "real_hat_runs.csv",
        ROOT / "results" / "lcbench" / "baselines" / "all_results.csv.gz",
        ROOT / "results" / "lcbench" / "final_runs.csv",
        ROOT / "results" / "hpobench" / "results_merged.csv",
        ROOT / "results" / "pathfinding" / "raw" / "pathfinding_full.txt",
        ROOT / "results" / "camera_ready" / "controller_study_runs.csv",
        ROOT / "requirements-camera-ready.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing reproducibility artifacts: {missing}")

    hat = pd.read_csv(required[4])
    checks = {
        "successful_runs": len(hat) == 340,
        "paired_blocks":
            hat[["instance", "seed"]].drop_duplicates().shape[0] == 340,
        "mean_final_error": math.isclose(
            float(hat["final_best_error"].mean()), -84.10807858635397, abs_tol=1e-12
        ),
        "std_final_error": math.isclose(
            float(hat["final_best_error"].std(ddof=1)),
            13.687862825715696,
            abs_tol=1e-12,
        ),
        "mean_cost": math.isclose(
            float(hat["total_cost"].mean()), 134018.08819695213, abs_tol=1e-9
        ),
        "std_cost": math.isclose(
            float(hat["total_cost"].std(ddof=1)),
            195574.98414735423,
            abs_tol=1e-9,
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Frozen-result verification failed: {checks}")
    print(json.dumps(checks, indent=2))


def statistics() -> None:
    run_script(ROOT / "lcbench_benchmark" / "camera_ready_statistics.py")


def figures() -> None:
    run_script(ROOT / "analysis" / "generate_figures.py")


def print_commands() -> None:
    commands = {
        "full_hat_lcbench_expensive": "python lcbench_benchmark/run_real_hat_full.py",
        "ablation_and_sensitivity_expensive": (
            "python lcbench_benchmark/camera_ready_studies.py "
            "--instances 3945 167104 168329 168908 189873 "
            "--seeds 0 1 2 3 4 --trials 60 --output-dir results/camera_ready"
        ),
        "hpobench_expensive": (
            "powershell -ExecutionPolicy Bypass -File "
            "hpobench_benchmark/run_all_datasets.ps1"
        ),
        "pathfinding_expensive": "python pathfinding_benchmark/pathfinding_benchmark.py --full",
        "statistics_saved_data_only": "python scripts/reproduce_paper.py stats",
        "figures_saved_data_only": "python scripts/reproduce_paper.py figures",
        "verify_saved_artifacts": "python scripts/reproduce_paper.py verify",
    }
    print(json.dumps(commands, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce HAT paper analyses without launching experiments by default."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify required files and frozen HAT values.")
    subparsers.add_parser("stats", help="Recompute statistics from saved final runs.")
    subparsers.add_parser("figures", help="Regenerate figures/tables from saved results.")
    subparsers.add_parser("commands", help="Print exact experiment and analysis commands.")
    subparsers.add_parser("analysis", help="Run saved-data statistics and figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "stats":
        statistics()
    elif args.command == "figures":
        figures()
    elif args.command == "commands":
        print_commands()
    elif args.command == "analysis":
        statistics()
        figures()


if __name__ == "__main__":
    main()
