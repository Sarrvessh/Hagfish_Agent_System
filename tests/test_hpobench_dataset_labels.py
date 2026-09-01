from __future__ import annotations

import pandas as pd
import pytest

from hpobench_benchmark.recover_dataset_labels import (
    EXPECTED_ALGORITHMS,
    EXPECTED_SEEDS,
    recover,
)
from hpobench_benchmark.run_all_hpobench import _append_standard_csv


def _historical_frame(dataset_count: int = 2) -> pd.DataFrame:
    rows = []
    for _ in range(dataset_count):
        for algorithm in EXPECTED_ALGORITHMS:
            for seed in EXPECTED_SEEDS:
                rows.extend(
                    [
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "cumulative_simulated_cost": 0.01,
                            "best_validation_error": 0.5,
                        },
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "cumulative_simulated_cost": 1.0,
                            "best_validation_error": 0.4,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_recovery_labels_complete_blocks() -> None:
    labeled, metadata = recover(_historical_frame(), ["dataset_a", "dataset_b"])

    assert labeled["dataset"].value_counts().to_dict() == {
        "dataset_a": 80,
        "dataset_b": 80,
    }
    assert metadata["run_blocks"] == 80
    assert all(metadata["validation"].values())


def test_recovery_rejects_run_order_mismatch() -> None:
    frame = _historical_frame()
    frame.loc[0:1, "seed"] = 9

    with pytest.raises(ValueError, match="Run order mismatch"):
        recover(frame, ["dataset_a", "dataset_b"])


def test_future_merges_write_dataset_column(tmp_path) -> None:
    source = tmp_path / "stage.csv"
    merged = tmp_path / "merged.csv"
    _historical_frame(dataset_count=1).iloc[:2].to_csv(source, index=False)

    appended = _append_standard_csv(source, merged, "australian")
    result = pd.read_csv(merged)

    assert appended == 2
    assert list(result.columns) == [
        "dataset",
        "algorithm",
        "seed",
        "cumulative_simulated_cost",
        "best_validation_error",
    ]
    assert result["dataset"].tolist() == ["australian", "australian"]
