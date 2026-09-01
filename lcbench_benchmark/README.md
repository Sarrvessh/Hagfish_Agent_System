# LCBench Evaluation

This directory contains two distinct execution paths:

1. `run_real_hat_full.py` and `camera_ready_studies.py` execute the complete
   `AdaptiveTrainer` planner-critic-memory controller used in the paper.
2. `hagfish_wrapper.py` is a superseded integration wrapper retained only for
   provenance. It does **not** implement Full HAT and is exposed by
   `benchmark_runner.py` only as `legacy_wrapper`.

## Data setup

Install the YAHPO data in `<repository>/yahpo_data`; no machine-specific path is
required.

```bash
git clone --filter=blob:none --sparse https://github.com/slds-lmu/yahpo_data.git yahpo_data
git -C yahpo_data sparse-checkout set benchmark_suites lcbench
```

## Full HAT

The following command is expensive. It covers all 34 instances, seeds 0-9,
and 200 trials, resuming from the saved result file when present.

```bash
python lcbench_benchmark/run_real_hat_full.py
```

Raw Full HAT runs are retained in
`results/lcbench/full_hat/real_hat_runs.csv`.

## Baselines

`benchmark_runner.py` provides Random Search, ASHA, BOHB, and DEHB. The default
algorithm set excludes the legacy wrapper.

```bash
python lcbench_benchmark/benchmark_runner.py \
  --scenario lcbench \
  --instances auto \
  --algorithms random_search asha bohb dehb \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-trials 200 \
  --output-dir benchmark_outputs/lcbench
```

The publication baseline trial stream is preserved at
`results/lcbench/baselines/all_results.csv.gz`.

## Saved-data statistics

```bash
python lcbench_benchmark/camera_ready_statistics.py
```

This command reads `results/lcbench/final_runs.csv`; it does not run the
benchmark.
