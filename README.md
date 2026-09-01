# Hagfish Adaptive Trainer

Hagfish Adaptive Trainer (HAT) is a lightweight planner-critic-memory controller
for adaptive budget allocation in resource-constrained hyperparameter
optimization. HAT is primarily a budget-control layer: configuration proposal
can remain external while HAT chooses episode-level population, iteration, and
elite budgets from observed quality and cost.

This repository contains the implementation, experiment entry points, saved raw
outputs, statistical analysis, and figure/table generation used by the paper
*Hagfish Adaptive Trainer: Resource-Constrained Hyperparameter Optimization and
Dynamic Budget Allocation*.

## Architecture

- **PlannerAgent** maintains and scores candidate budget states.
- **CriticAgent** labels outcomes as improved, saturated, or stagnated.
- **AgentMemory** records elite states, penalized regimes, stagnation, and history.
- **AdaptiveTrainer** exposes the closed loop through `plan()` and `observe()`.
- The **execution layer** remains external and evaluates the selected budget.

The complete controller is implemented in `adaptive_trainer/`. The legacy
LCBench integration wrapper is retained only for provenance and is explicitly
marked as not being Full HAT.

## Repository Structure

```text
adaptive_trainer/       Core HAT planner, critic, memory, and controller
lcbench_benchmark/      LCBench/YAHPO runners, ablations, sensitivity, statistics
hpobench_benchmark/     HPOBench experiment and analysis entry points
pathfinding_benchmark/  Pathfinding benchmark implementation
analysis/               Saved-data figure and table generation
scripts/                Safe paper-reproduction entry point
results/
  lcbench/              Full HAT raw runs, baseline raw archive, final paired runs
  hpobench/              Final HPOBench outputs and statistics
  pathfinding/           Final pathfinding raw output
  camera_ready/          Final statistical, ablation, and sensitivity tables
tests/                   Core behavior and API tests
```

## Installation

The paper analyses were executed with Python 3.11.9. Exact package versions are
recorded in `requirements-camera-ready.txt`.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-camera-ready.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-camera-ready.txt
```

LCBench uses YAHPO Gym's separately distributed benchmark data. Install it in a
repository-local `yahpo_data/` directory; the directory is ignored because it is
a reconstructable third-party cache.

```bash
git clone --filter=blob:none --sparse https://github.com/slds-lmu/yahpo_data.git yahpo_data
git -C yahpo_data sparse-checkout set benchmark_suites lcbench
```

## Reproducing Paper Results

The reproduction entry point never starts an expensive experiment by default.

```bash
python scripts/reproduce_paper.py --help
python scripts/reproduce_paper.py verify
python scripts/reproduce_paper.py stats
python scripts/reproduce_paper.py figures
python scripts/reproduce_paper.py commands
```

- `verify` checks required artifacts and the frozen 340-run Full HAT values.
- `stats` recomputes paired tests from `results/lcbench/final_runs.csv`.
- `figures` regenerates figures and tables from saved results.
- `commands` prints the exact experiment commands without running them.
- `analysis` runs only the saved-data statistics and figure stages.

### Full HAT LCBench

The expensive 34-instance, ten-seed, 200-trial controller run is:

```bash
python lcbench_benchmark/run_real_hat_full.py
```

Its saved raw output is
`results/lcbench/full_hat/real_hat_runs.csv`. The experiment calls the real
`AdaptiveTrainer.plan()`/`observe()` loop with `PlannerAgent`, `CriticAgent`, and
`AgentMemory`. Baseline trial-level outputs are retained in compressed form at
`results/lcbench/baselines/all_results.csv.gz`.

### Ablation and Sensitivity

```bash
python lcbench_benchmark/camera_ready_studies.py \
  --instances 3945 167104 168329 168908 189873 \
  --seeds 0 1 2 3 4 \
  --trials 60 \
  --output-dir results/camera_ready
```

This command is expensive. The final saved outputs are already present in
`results/camera_ready/`.

### HPOBench

```bash
powershell -ExecutionPolicy Bypass -File hpobench_benchmark/run_all_datasets.ps1
```

New runs write the dataset identifier directly into the merged trajectory. The
historical merged file predates that schema, but its five contiguous dataset
blocks are recoverable from the recorded runner order and complete
4-algorithm x 10-seed grid. Reproduce the labeled artifact and AUC analysis with:

```bash
python hpobench_benchmark/recover_dataset_labels.py
python hpobench_benchmark/analyze_benchmark_results.py \
  --input-csv results/hpobench/results_merged_labeled.csv \
  --output-dir results/hpobench/statistics_auc \
  --ranking-metric auc
```

The recovery writes `results_merged_labeled.csv` and a validation record with
the source SHA-256, block boundaries, and grid checks to
`results/hpobench/dataset_label_recovery.json`. The original merged file is
preserved unchanged.

### Pathfinding

```bash
python pathfinding_benchmark/pathfinding_benchmark.py --full
```

The final raw benchmark output used by the paper is retained at
`results/pathfinding/raw/pathfinding_full.txt`.

## Main Reported Results

HPOBench uses five datasets (Australian, Blood Transfusion, Car, Phoneme, and
Vehicle), four algorithms, and seeds 0-9. AUC is computed separately for every
dataset-seed trajectory and then ranked within each dataset.

| Method | Mean dataset log-cost AUC (lower is better) | Average rank |
|---|---:|---:|
| SHA | 0.2321 | 1.2 |
| Hyperband | 0.2338 | 2.4 |
| HAT | 0.2354 | 2.8 |
| TPE | 0.2364 | 3.6 |

The five-dataset Friedman test gives chi-square(3) = 9.000 and `p = 0.0293`.
No HAT-versus-baseline Wilcoxon comparison remains significant after Holm
correction, so the paper does not claim HPOBench superiority.

LCBench uses 34 instances, seeds 0-9, 200 trials per run, and 340 paired
instance-seed blocks. ASHA uses the corrected Syne Tune scheduler lifecycle.
The archived evolutionary result is labeled DEHB-style because it is not an
execution of the official DEHB package. Incomplete BOHB recomputations are not
reported.

| Method | Mean final transformed objective (lower is better) | Mean simulated cost |
|---|---:|---:|
| DEHB-style | -88.6874 | 217,640.07 |
| HAT | -84.1081 | 134,018.09 |
| ASHA | -81.9679 | 12,048.95 |
| Random Search | -47.3619 | 1,945.92 |

HAT used 38.4% less mean simulated cost than DEHB-style but had worse final
quality. Against corrected ASHA, HAT had better final quality at higher cost.
The result is a set of quality-cost operating points, not a universal
superiority claim.

The Full HAT standard deviations are 13.6879 for the final transformed objective and 195,574.98 for
simulated cost, with 340/340 successful runs. The saved-data Friedman test over
HAT, DEHB-style, and ASHA gives chi-square(2) = 412.947 and
`p = 2.14e-90`.

## Reproducibility Record

- Python: 3.11.9
- NumPy: 1.26.4
- pandas: 2.2.3
- SciPy: 1.15.3
- Matplotlib: 3.10.3
- ConfigSpace: 0.6.1
- Syne Tune: 0.15.0
- YAHPO Gym: 1.0.2
- ONNX Runtime: 1.23.2
- Optuna: 4.6.0
- LCBench seeds: 0-9
- LCBench instances: 34
- Full HAT trials per run: 200

## Citation

Publication metadata will be added after publication. Until then, use the
repository metadata in `CITATION.cff` without inventing a DOI or proceedings
record.

## License

The software is released under the MIT License. See `LICENSE`.
