"""Run the Full HAT controller on all 34 saved LCBench blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from camera_ready_statistics import holm_adjust, load_final_runs, rank_biserial
from camera_ready_studies import run_controller


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "lcbench" / "baselines"
OUTPUT = ROOT / "results" / "lcbench" / "full_hat"
FINAL_RUNS = ROOT / "results" / "lcbench" / "final_runs.csv"
RAW_PATH = OUTPUT / "real_hat_runs.csv"


def main() -> None:
    metadata = json.loads((ARCHIVE / "metadata.json").read_text(encoding="utf-8"))
    instances = [str(value) for value in metadata["instances"]]
    seeds = list(range(10))
    if len(instances) != 34:
        raise RuntimeError(f"Expected 34 LCBench instances, found {len(instances)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if RAW_PATH.exists():
        rows = pd.read_csv(RAW_PATH).to_dict("records")
    completed = {(str(row["instance"]), int(row["seed"])) for row in rows}

    for instance in instances:
        for seed in seeds:
            if (instance, seed) in completed:
                continue
            row = run_controller(
                instance=instance,
                seed=seed,
                trials=200,
                alpha=5e-4,
                variant="Full HAT",
                flags={},
            )
            rows.append(row)
            pd.DataFrame(rows).sort_values(["instance", "seed"]).to_csv(RAW_PATH, index=False)

    hat = pd.DataFrame(rows).sort_values(["instance", "seed"])
    if len(hat) != 340:
        raise RuntimeError(f"Expected 340 successful HAT runs, found {len(hat)}")

    archived = pd.read_csv(FINAL_RUNS)
    archived["algorithm"] = archived["algorithm"].str.lower().replace({"hagfish": "hat"})
    archived = archived[archived["algorithm"].isin(["dehb_style", "asha"])]
    hat_for_tests = hat[["instance", "seed", "final_best_error", "total_cost"]].copy()
    hat_for_tests["algorithm"] = "hat"
    combined = pd.concat([archived, hat_for_tests], ignore_index=True)
    matrix = combined.pivot(index=["instance", "seed"], columns="algorithm", values="final_best_error")
    matrix = matrix[["hat", "dehb_style", "asha"]].dropna()

    comparisons = []
    for baseline in ["dehb_style", "asha"]:
        x = matrix["hat"].to_numpy(dtype=float)
        y = matrix[baseline].to_numpy(dtype=float)
        result = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        comparisons.append(
            {
                "baseline": baseline.upper(),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "effect_size_rank_biserial_hat_minus_baseline": rank_biserial(x, y),
            }
        )
    adjusted = holm_adjust(item["p_value"] for item in comparisons)
    for item, value in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = value

    friedman = stats.friedmanchisquare(
        matrix["hat"].to_numpy(),
        matrix["dehb_style"].to_numpy(),
        matrix["asha"].to_numpy(),
    )
    report = {
        "mean_final_quality": float(hat["final_best_error"].mean()),
        "std_final_quality": float(hat["final_best_error"].std(ddof=1)),
        "mean_simulated_cost": float(hat["total_cost"].mean()),
        "std_simulated_cost": float(hat["total_cost"].std(ddof=1)),
        "successful_runs": int(len(hat)),
        "pairwise_final_quality": comparisons,
        "friedman_final_quality": {
            "n_blocks": int(len(matrix)),
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
        },
    }
    (OUTPUT / "requested_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
