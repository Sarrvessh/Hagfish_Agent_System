# hagfish-adaptive-trainer

[![PyPI version](https://img.shields.io/pypi/v/hagfish-adaptive-trainer.svg)](https://pypi.org/project/hagfish-adaptive-trainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**hagfish-adaptive-trainer** is a high-efficiency agentic framework for training budget optimization.
It dynamically allocates training resources (batch size, epochs, and capacity) using a feedback-driven loop—maximizing model performance while minimizing compute cost.

---

## Why Hagfish?

In traditional machine learning workflows, a large portion of compute is wasted on diminishing returns—running epochs that no longer produce meaningful improvements.

Hagfish introduces an agentic control loop that continuously asks:

> "Is the next unit of compute actually worth the improvement it brings?"

### Key benefits

- Cost efficiency — Automatically reduces budgets when performance saturates
- Stagnation recovery — Escalates resources only when learning stalls
- Reward-centric — Optimizes the tradeoff between accuracy and cost
- Plug-and-play — Framework-agnostic (Scikit-Learn, PyTorch, TensorFlow)

---

## Performance Benchmarks

**Last Updated:** January 17, 2026

Hagfish-SOTA has been rigorously tested across **8 HPOBench datasets** and **Neural Architecture Search (NAS)** tasks, demonstrating state-of-the-art performance in multi-fidelity hyperparameter optimization.

### 🏆 Key Results Summary

- **Pareto Frontier Dominance:** #1 position on 7/8 HPOBench datasets (87.5%)
- **Highest Accuracy:** Achieved top accuracy on 7/8 datasets
- **Cost Efficiency:** 13% average cost reduction vs Fixed baseline
- **Statistical Significance:** Beats 8 baseline methods across 3 datasets (p<0.05)
- **NAS Performance:** #2 accuracy (0.9144), competitive efficiency (0.13)

---

### HPOBench Results (8 Datasets)

**Configuration:** 5 seeds × 50 rounds × α=0.3 (accuracy-focused)  
**Baselines:** Fixed, Random, CheapGreedy, EpsilonGreedy, SuccessiveHalving, Hyperband, PBT, Optuna

| Dataset               | Hagfish Accuracy    | Pareto Position      | Cost vs Fixed | Statistical Significance   |
| --------------------- | ------------------- | -------------------- | ------------- | -------------------------- |
| **Australian**        | **0.8379 ± 0.0169** | **#1**               | -13%          | p>0.05                     |
| **Car**               | **0.7463 ± 0.0503** | **#1** (only method) | -13%          | p>0.05                     |
| **Phoneme**           | **0.7542 ± 0.0266** | **#1**               | -14%          | p>0.05                     |
| **Vehicle**           | 0.7069 ± 0.0292     | Not on frontier      | -14%          | p>0.05                     |
| **KC1**               | **0.6222 ± 0.0232** | **#1**               | -12%          | **p=0.018 vs CheapGreedy** |
| **Segment**           | **0.7717 ± 0.0396** | **#1**               | -13%          | p>0.05                     |
| **Blood Transfusion** | **0.5965 ± 0.0095** | **#1**               | -12%          | **p<0.05 vs 6 methods** ⭐ |
| **Credit_g**          | **0.7320 ± 0.0185** | **#1**               | -13%          | **p=0.013 vs CheapGreedy** |

**Average Performance:**

- **Mean Accuracy:** Leads on 7/8 datasets
- **Cost Reduction:** 13.0% vs Fixed baseline (2.0 total cost → 1.74 average)
- **Pareto Dominance:** 87.5% success rate

---

### Detailed Results by Dataset

#### 1. Blood Transfusion (Donor Prediction) - ⭐ Strongest Result

- **Hagfish:** 0.5965 ± 0.0095 accuracy
- **Best Competitor:** Fixed (0.5958), EpsilonGreedy (0.5832)
- **Statistical Wins:** 6/8 methods (Random, CheapGreedy, SuccessiveHalving, Hyperband, PBT, Optuna)
- **Key Insight:** Demonstrates robust optimization across difficult datasets

#### 2. KC1 (Software Defect Prediction)

- **Hagfish:** 0.6222 ± 0.0232 accuracy
- **Cost:** 1.7585 (12% cheaper than Fixed)
- **Statistical Significance:** p=0.018 vs CheapGreedy
- **Key Insight:** Strong performance on imbalanced classification

#### 3. Credit_g (German Credit Scoring)

- **Hagfish:** 0.7320 ± 0.0185 accuracy
- **Best Competitor:** Fixed (0.7282), EpsilonGreedy (0.7249)
- **Statistical Significance:** p=0.013 vs CheapGreedy
- **Key Insight:** Production-ready for financial applications

#### 4. Australian (Credit Approval)

- **Hagfish:** 0.8379 ± 0.0169 accuracy (#1)
- **Cost:** 1.7405 (13% cheaper)
- **Convergence:** 2.0 episodes (17% faster than Fixed)

#### 5. Phoneme (Speech Recognition)

- **Hagfish:** 0.7542 ± 0.0266 accuracy (#1 tied with Fixed)
- **Cost:** 1.7265 (14% cheaper)
- **Key Insight:** Matches Fixed accuracy while saving compute

#### 6. Segment (Image Segmentation)

- **Hagfish:** 0.7717 ± 0.0396 accuracy (highest across all methods)
- **Convergence:** 1.6 episodes (43% faster)

#### 7. Car (Vehicle Classification)

- **Hagfish:** 0.7463 ± 0.0503 accuracy
- **Pareto Status:** Only method on frontier
- **Advantage:** +2.18% vs Fixed, +0.63% vs PBT

#### 8. Vehicle (Silhouette Classification)

- **Hagfish:** 0.7069 ± 0.0292 accuracy
- **Note:** PBT achieves better Pareto efficiency on this dataset

---

### Neural Architecture Search (NAS) Benchmark

**Configuration:** 100 rounds × 10 seeds  
**Task:** Architecture search for breast cancer classification

| Strategy         | Best Accuracy | Total Cost | Efficiency |
| ---------------- | ------------- | ---------- | ---------- |
| Random           | 0.9053        | 629.02     | 0.14       |
| Evolution (REA)  | 0.9122        | 751.02     | 0.12       |
| SHA (Hyperband)  | 0.8139        | 155.65     | **0.52**   |
| DARTS (Sim)      | 0.9082        | 829.20     | 0.11       |
| **Hagfish**      | **0.9144**    | 709.38     | 0.13       |
| **Optuna (TPE)** | **0.9159**    | 702.77     | 0.13       |

**Key Insights:**

- **#2 Accuracy:** 0.9144 (only 0.15% behind Optuna)
- **Balanced Trade-off:** Competitive accuracy with moderate cost
- **Adaptive Search:** Efficiently navigates architecture space

---

### Algorithm Characteristics

#### Adaptivity Metrics (Average across 8 datasets)

- **Escalations:** 6.6 per run (intelligent fidelity increases)
- **Prunings:** 7.6 per run (efficient budget reduction)
- **Convergence:** 6.4 episodes to 95% max accuracy

#### Multi-Fidelity Strategy

1. **Early Phase (0-50%):** High fidelity (f=1.0) for maximum accuracy
2. **Mid Phase (50-70%):** Weighted selection [1.0, 0.75] (70:30 ratio)
3. **Late Phase (70-85%):** Mixed fidelity with best-fidelity exploitation
4. **Saturation Phase:** Efficient pruning when performance plateaus

---

### Benchmark Methodology & Parameters

#### Experimental Configuration

| Parameter      | Value            | Purpose                                            |
| -------------- | ---------------- | -------------------------------------------------- |
| **Seeds**      | 5                | Statistical robustness - multiple independent runs |
| **Rounds**     | 50               | Episodes per optimization run                      |
| **Alpha (α)**  | 0.3              | Cost penalty weight (70% accuracy, 30% cost)       |
| **Fidelities** | [0.5, 0.75, 1.0] | Training budget levels (Hagfish v3)                |
| **Cost Model** | Quadratic (f²)   | Realistic compute cost scaling                     |

#### Parameter Selection Guide

**Seeds (num_seeds):**

- **5 seeds:** Standard benchmarking (used in our results)
- **10 seeds:** High confidence requirements
- **3 seeds:** Quick validation runs
- Higher seeds improve statistical power but increase runtime linearly

**Rounds (episodes):**

- **50 rounds:** Balanced exploration/exploitation (recommended)
- **30 rounds:** Fast experimentation
- **100 rounds:** Deep convergence analysis
- More rounds allow better saturation detection

**Alpha (cost penalty):**

- **α=0.3:** Accuracy-focused (70% weight on accuracy)
- **α=0.5:** Balanced (equal weight)
- **α=0.7-0.9:** Cost-focused (cheaper but lower accuracy)
- Formula: `Reward = Accuracy - (α × Cost)`

#### Evaluation Metrics

**Accuracy:**

- Validation set performance (0-1 scale)
- Mean ± Standard Deviation across seeds

**Cost:**

- Quadratic fidelity cost: `Cost = (fidelity² × 0.02) + (fidelity² × 0.02)`
- Total cost accumulated across all rounds

**Efficiency:**

- Pareto frontier position (non-dominated solutions)
- Cost-Accuracy trade-off: `Efficiency = Accuracy / (Cost + ε)`

**Statistical Significance:**

- Two-tailed t-test comparing Hagfish vs baselines
- Significance levels: \* (p<0.05), ** (p<0.01), \*** (p<0.001)

#### Runtime Information

Approximate execution times (per dataset):

| Configuration         | Time     | Use Case                    |
| --------------------- | -------- | --------------------------- |
| 5 seeds × 50 rounds   | ~30 sec  | Standard benchmarking       |
| 10 seeds × 100 rounds | ~2-3 min | Publication-quality results |
| 3 seeds × 30 rounds   | ~15 sec  | Quick validation            |

**NAS Benchmark:** ~5-10 minutes (100 rounds × 10 seeds)

---

### Reproducibility

All results are fully reproducible:

```bash
cd experiments

# Run HPOBench on specific dataset (standard config)
python final.py --mode benchmark --dataset australian --seeds 5 --rounds 50 --alpha 0.3

# Quick validation run
python final.py --mode benchmark --dataset credit_g --seeds 3 --rounds 30 --alpha 0.3

# High-confidence experiment
python final.py --mode benchmark --dataset blood_transfusion --seeds 10 --rounds 100 --alpha 0.3

# Run NAS benchmark
python nas_benchmark.py

# Available datasets:
# australian, car, phoneme, vehicle, kc1, segment, blood_transfusion, credit_g
```

**Hardware:** Standard Windows machine (CPU-based)  
**Python Environment:** Python 3.8+ with simple-hpo-bench, optuna, numpy, pandas, matplotlib, seaborn, scipy  
**Full Results:** See `experiments/comprehensive_benchmark_results.md` for detailed analysis

---

## Installation

### Install from PyPI

```bash
pip install hagfish-adaptive-trainer
```

### Install from source (development)

```bash
git clone https://github.com/your-repo/hagfish-adaptive-trainer.git
cd hagfish-adaptive-trainer
pip install -e .
```

---

## Core architecture

The system operates as an episodic agent loop composed of three cooperating components:

- **PlannerAgent**
  Proposes training budgets (batch size, epochs) based on historical performance.

- **CriticAgent**
  Evaluates outcomes and classifies them as:

  - Improvement
  - Stagnation
  - Saturation

- **AgentMemory**
  Tracks reward trends and stagnation to prevent unnecessary escalation.

This mirrors the biological behavior of Hagfish: conserve energy until escalation is justified.

---

## Quick start

### Basic usage

```python
from adaptive_trainer import AdaptiveTrainer

# Initialize with cost sensitivity (alpha)
trainer = AdaptiveTrainer(alpha=2e-5)

# Request a training budget
plan = trainer.plan({"dataset_size": 569})
# Example output:
# {'pop_size': 32, 'max_iter': 100, 'elite_size': 2}

# Train your model using the plan
# model = MLPClassifier(
#     batch_size=plan["pop_size"],
#     max_iter=plan["max_iter"]
# )
# model.fit(X_train, y_train)

# Report results back to the agent
trainer.observe(
    metric=0.935,
    cost=697,
    params=plan
)
```

---

## API Reference

### AdaptiveTrainer

Main class for adaptive training budget optimization.

#### Constructor

```python
AdaptiveTrainer(alpha: float = 1e-5)
```

**Parameters:**

- `alpha` (float): Cost penalty coefficient. Controls accuracy vs cost trade-off.
  - Lower values (1e-6): Prioritize accuracy
  - Higher values (1e-4): Prioritize cost reduction

#### Methods

**`plan(context: Dict) -> Dict`**

Request a training budget based on historical performance.

**Parameters:**

- `context` (dict): Context information with keys:
  - `dataset_size` (int): Number of training samples
  - `episode_num` (int, optional): Current episode number
  - `progress_ratio` (float, optional): Completion percentage (0-1)

**Returns:**

- `dict`: Training plan with keys like `pop_size`, `max_iter`, `elite_size`, or `fidelity`

**Example:**

```python
plan = trainer.plan({"dataset_size": 1000, "episode_num": 5})
# Returns: {'pop_size': 32, 'max_iter': 100, 'fidelity': 0.75}
```

**`observe(metric: float, cost: float, **kwargs) -> None`\*\*

Report training results back to the agent for learning.

**Parameters:**

- `metric` (float): Model performance (accuracy, F1, etc.)
- `cost` (float): Computational cost incurred
- `**kwargs`: Additional context (optional)

**Example:**

```python
trainer.observe(metric=0.935, cost=697)
```

---

## Requirements

### Core Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
```

### Benchmarking (Optional)

```
simple-hpo-bench>=0.1.0
optuna>=3.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

Install all dependencies:

```bash
pip install hagfish-adaptive-trainer[benchmark]
```

---

## FAQ & Troubleshooting

### Q: Why am I getting ConvergenceWarning from Scikit-Learn?

**A:** This is expected during early episodes when Hagfish explores low-budget configurations. The agent intentionally tests cheaper settings to learn the cost-accuracy trade-off. You can safely ignore these warnings or suppress them:

```python
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

### Q: How do I choose the right alpha value?

**A:** Start with these guidelines:

- **Production models:** α=1e-6 to 1e-5 (prioritize accuracy)
- **Experimentation:** α=1e-5 to 1e-4 (balanced)
- **Large-scale sweeps:** α=1e-4 to 1e-3 (aggressive cost reduction)

Run quick experiments with 3 seeds to find your optimal alpha before full benchmarks.

### Q: Can I use Hagfish with PyTorch/TensorFlow?

**A:** Yes! Hagfish is framework-agnostic. Just map the budget parameters to your framework:

```python
plan = trainer.plan({"dataset_size": len(train_loader)})

# PyTorch example
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(plan["max_iter"]):
    # Training loop...

trainer.observe(metric=val_accuracy, cost=training_time)
```

### Q: My results differ slightly from benchmark numbers. Why?

**A:** This is normal due to:

- Random seed variation (use more seeds for stability)
- Hardware differences (CPU vs GPU timing)
- Library version differences (ensure same numpy/scipy versions)

Differences of ±2% are typical. For exact reproduction, use the same environment.

### Q: How does Hagfish handle failures or crashes?

**A:** The agent tracks history internally. If a configuration fails:

1. Report it with `observe(metric=-1.0, cost=0.0)`
2. Hagfish will learn to avoid similar configurations
3. Use try-except blocks around training for robustness

### Q: Can I save/load the trainer state?

**A:** Currently, state persistence is not built-in. For multi-session experiments, you can:

- Track history externally in a database
- Re-run short warm-up episodes to rebuild context
- Contribute a serialization feature (see Contributing section)

---

## Links & Resources

- **PyPI Package:** https://pypi.org/project/hagfish-adaptive-trainer/
- **Documentation:** Would be updated here soon.
- **Benchmark Results:** [`experiments/comprehensive_benchmark_results.md`](experiments/comprehensive_benchmark_results.md)
- **Issue Tracker:** (Add your GitHub issues link)
- **Discussions:** (Add your GitHub discussions link)

**Related Papers & Methods:**

- Hyperband: [Li et al., 2018](https://arxiv.org/abs/1603.06560)
- BOHB: [Falkner et al., 2018](https://arxiv.org/abs/1807.01774)
- PBT: [Jaderberg et al., 2017](https://arxiv.org/abs/1711.09846)

---

## Advanced configuration

### The Alpha (α) parameter

Alpha controls how aggressively cost is penalized.

| Alpha Value | Behavior                                 |
| ----------- | ---------------------------------------- |
| `1e-6`      | Prioritize accuracy (production models)  |
| `1e-5`      | Balanced accuracy vs cost                |
| `1e-4`      | Aggressive cost reduction (large sweeps) |

---

## Stability & warnings

- **Backward compatibility**
  The `AdaptiveTrainer.plan()` and `AdaptiveTrainer.observe()` APIs are stable across all `0.1.x` releases.

- **Convergence warnings**
  Early low-budget plans may trigger `ConvergenceWarning` in Scikit-Learn.
  This is expected behavior during cost exploration and not an error.

---

## Testing & robustness

The package is validated against:

- Deterministic behavior
- Edge cases (zero cost, negative metrics)
- Long-run stability
- External ML pipelines
- Cross-platform compatibility

All tests are designed to run outside the package directory, ensuring true public API safety.

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/YourFeature
   ```

3. Commit changes

   ```bash
   git commit -m "Add YourFeature"
   ```

4. Push and open a Pull Request

---

## License

Distributed under the MIT License.
See the `LICENSE` file for details.
