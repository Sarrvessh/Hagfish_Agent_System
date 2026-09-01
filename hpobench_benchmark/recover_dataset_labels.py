"""Recover dataset labels for the historical merged HPOBench trajectory file.

The original overnight runner appended one complete dataset at a time but the
historical merge schema omitted the dataset column.  Recovery is deterministic
because each dataset contributes the same ordered set of 40 contiguous runs:
four algorithms by ten seeds.  The dataset order is the successful subset of
``run_all_datasets.ps1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASETS = [
    "australian",
    "car",
    "phoneme",
    "vehicle",
    "blood_transfusion",
]
EXPECTED_ALGORITHMS = ["sha", "hyperband", "tpe", "hagfish"]
EXPECTED_SEEDS = list(range(10))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "results" / "hpobench" / "results_merged.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "hpobench" / "results_merged_labeled.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "results" / "hpobench" / "dataset_label_recovery.json",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    return parser.parse_args()


def contiguous_run_ids(frame: pd.DataFrame) -> pd.Series:
    starts = (frame["algorithm"] != frame["algorithm"].shift()) | (
        frame["seed"] != frame["seed"].shift()
    )
    return starts.cumsum().astype(int) - 1


def recover(frame: pd.DataFrame, datasets: list[str]) -> tuple[pd.DataFrame, dict]:
    required = {
        "algorithm",
        "seed",
        "cumulative_simulated_cost",
        "best_validation_error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "dataset" in frame.columns:
        raise ValueError("Input already contains a dataset column")

    frame = frame.copy()
    frame["_run_id"] = contiguous_run_ids(frame)
    runs = (
        frame.groupby("_run_id", sort=True)
        .agg(
            algorithm=("algorithm", "first"),
            seed=("seed", "first"),
            first_cost=("cumulative_simulated_cost", "first"),
            rows=("algorithm", "size"),
        )
        .reset_index()
    )

    expected_pairs = [
        (algorithm, seed)
        for algorithm in EXPECTED_ALGORITHMS
        for seed in EXPECTED_SEEDS
    ]
    runs_per_dataset = len(expected_pairs)
    if len(runs) != len(datasets) * runs_per_dataset:
        raise ValueError(
            f"Expected {len(datasets) * runs_per_dataset} contiguous runs, "
            f"found {len(runs)}"
        )

    labels: dict[int, str] = {}
    block_records = []
    for block_index, dataset in enumerate(datasets):
        start = block_index * runs_per_dataset
        stop = start + runs_per_dataset
        block = runs.iloc[start:stop]
        observed_pairs = list(zip(block["algorithm"], block["seed"]))
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"Run order mismatch in dataset block {block_index}: "
                f"expected {expected_pairs}, found {observed_pairs}"
            )
        if not (block["first_cost"].sub(0.01).abs() < 1e-12).all():
            raise ValueError(f"Dataset block {block_index} has non-reset starting cost")
        for run_id in block["_run_id"]:
            labels[int(run_id)] = dataset
        block_records.append(
            {
                "block_index": block_index,
                "dataset": dataset,
                "run_id_start": int(block["_run_id"].min()),
                "run_id_end": int(block["_run_id"].max()),
                "trajectory_rows": int(block["rows"].sum()),
                "algorithms": EXPECTED_ALGORITHMS,
                "seeds": EXPECTED_SEEDS,
            }
        )

    frame.insert(0, "dataset", frame["_run_id"].map(labels))
    if frame["dataset"].isna().any():
        raise ValueError("At least one trajectory row could not be labeled")
    frame = frame.drop(columns="_run_id")

    metadata = {
        "method": "contiguous dataset blocks from overnight runner append order",
        "datasets": datasets,
        "expected_algorithms": EXPECTED_ALGORITHMS,
        "expected_seeds": EXPECTED_SEEDS,
        "rows": int(len(frame)),
        "run_blocks": int(len(runs)),
        "validation": {
            "complete_algorithm_seed_grid_per_dataset": True,
            "all_runs_reset_at_cumulative_cost_0_01": True,
            "no_unlabeled_rows": True,
        },
        "blocks": block_records,
    }
    return frame, metadata


def main() -> None:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    runner_path = Path(__file__).with_name("run_all_datasets.ps1")
    frame = pd.read_csv(args.input)
    labeled, metadata = recover(frame, list(args.datasets))
    root = Path(__file__).resolve().parents[1]

    def _repo_relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return resolved.as_posix()

    metadata["source_file"] = _repo_relative(args.input)
    metadata["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    metadata["runner_file"] = _repo_relative(runner_path)
    metadata["runner_sha256"] = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    metadata["recovery_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    metadata["dataset_name_normalization"] = {
        "blood-transfusion": "blood_transfusion"
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # recovery_script_sha256 is recorded after this file is read below.
    print(f"Wrote {len(labeled):,} labeled rows to {args.output}")
    print(f"Wrote provenance metadata to {args.metadata}")


if __name__ == "__main__":
    main()
