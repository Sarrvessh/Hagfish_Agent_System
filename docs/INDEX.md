# Hagfish-SOTA Documentation Index

Welcome to the Hagfish-SOTA documentation! This guide will help you find the information you need.

---

## 📚 Getting Started

### Essential Guides

- **[README](../README.md)** - Main project overview and quick start
- **[Quick Start](QUICK_START.md)** - Get running in 5 minutes
- **[API Reference](API_REFERENCE.md)** - Complete API documentation

---

## 🎯 Core Concepts

### Technical Specifications

- **[Cost Model Specification](COST_MODEL_SPECIFICATION.md)** - Quadratic cost function: `Cost(f) = 0.04 × f²`
- **[Multi-Fidelity Strategy](../README.md#how-it-works)** - Adaptive budget allocation (0.2-1.0 fidelity)
- **[Agentic Control Loop](../README.md#the-agentic-control-loop)** - Planner, Critic, Memory architecture

---

## 📊 Benchmark Results

### Main Results

- **[Comprehensive Benchmark Results](../experiments/comprehensive_benchmark_results.md)** - Detailed 8-dataset analysis
  - HPOBench: 6/8 datasets on Pareto frontier
  - 11.9% average cost savings
  - p < 0.05 wins on 4/8 datasets

### Specialized Benchmarks

- **[NAS Benchmark Specification](NAS_BENCHMARK_SPECIFICATION.md)** - Neural Architecture Search
  - #2 accuracy (0.9144)
  - Competitive with Optuna (0.9159)
- **[Baseline Implementations](BASELINE_IMPLEMENTATIONS.md)** - All 8 comparison methods
  - Fixed, Random, CheapGreedy, EpsilonGreedy
  - SuccessiveHalving, Hyperband, PBT, Optuna

---

## 🏆 State-of-the-Art Comparison

### Modern Methods (2021-2025)

- **[SOTA Comparison Summary](issues/ISSUE_10_COMPLETE.md)** - Quick reference
  - **2.7× faster** than DEHB (~10 episodes)
  - **4.9× faster** than SMAC3 (~18 episodes)
  - **2.2× faster** than Optuna 4.6 (~8 episodes)

- **[Convergence Analysis](issues/ISSUE_8_COMPLETE.md)** - Detailed convergence study
  - **3.67 ± 2.31 episodes** to 95% accuracy
  - **#2 ranking** (after Fixed, which has no adaptivity)
  - **#1 AUC** (best overall trajectory)

- **[DEHB/SMAC3 Literature Comparison](issues/ISSUE_10_LITERATURE_COMPARISON.md)** - Published benchmarks
  - DEHB: 0.862 accuracy on Australian (Awad et al., NeurIPS 2021)
  - SMAC3: ~0.85 estimated (Lindauer et al., JMLR 2022)
  - Hagfish: 0.842 accuracy with 11.9% lower cost

---

## 🔬 Research Deep Dives

### Issue Resolution Archive

Comprehensive analyses from development:

#### Convergence & Performance

- **[Issue #8: Convergence Evidence](issues/ISSUE_8_COMPLETE.md)** ✅ Validated 3.67 episodes
  - Statistical tests vs 8 baselines
  - Convergence curves and trajectory analysis
  - AUC superiority: 0.795 ± 0.052 (#1 overall)

#### Cost Efficiency

- **[Issue #5: Cost Analysis](issues/ISSUE_5_COMPLETE.md)** ✅ 11.9% average savings
  - Dataset-specific cost breakdowns
  - Pareto frontier analysis
  - Adaptive vs Fixed comparison

#### Statistical Validation

- **[Issue #4: Statistical Tests](issues/ISSUE_4_COMPLETE.md)** ✅ 4/8 significant wins
  - Two-tailed t-tests (p < 0.05)
  - Effect sizes (Cohen's d)
  - Power analysis

#### Hyperparameter Tuning

- **[Issue #6: Alpha Sensitivity](issues/ISSUE_6_COMPLETE.md)** ✅ Optimal α = 0.3
  - Alpha ablation study (0.1-0.9)
  - Accuracy vs cost tradeoffs
  - Pareto efficiency analysis

#### Modern Comparisons

- **[Issue #10: SOTA Methods](issues/ISSUE_10_COMPLETE.md)** ✅ 2.7× faster than DEHB
  - DEHB, SMAC3, Optuna 4.6 benchmarks
  - Literature-based methodology
  - LaTeX templates for papers

#### Implementation Details

- **[Issue #3: Multi-Fidelity](issues/ISSUE_3_COMPLETE.md)** ✅ 3-level fidelity system
  - Fidelity levels: [0.5, 0.75, 1.0]
  - Adaptive selection strategy
  - Phase-based allocation

#### Baseline Correctness

- **[Issue #2: Baseline Validation](issues/ISSUE_2_RESOLUTION.md)** ✅ All baselines verified
  - Hyperband, PBT, Optuna implementations
  - Cross-validation with published results
  - Reproducibility checks

#### Framework Integration

- **[Issue #1: API Design](issues/ISSUE_1_COMPLETE.md)** ✅ Framework-agnostic API
  - Scikit-Learn, PyTorch, TensorFlow examples
  - plan() / observe() interface
  - Plug-and-play architecture

---

## 📖 Usage Examples

### Framework-Specific Integration

#### Scikit-Learn

```python
from adaptive_trainer import AdaptiveTrainer
from sklearn.neural_network import MLPClassifier

trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train)})
    model = MLPClassifier(max_iter=plan['max_iter'],
                          batch_size=plan['batch_size'])
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    trainer.observe(metric=accuracy, cost=plan['max_iter'] * plan['fidelity']**2)
```

#### PyTorch

```python
import torch
from adaptive_trainer import AdaptiveTrainer

trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(train_loader)})
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(plan['max_iter']):
        train_one_epoch(model, optimizer, train_loader)

    accuracy = evaluate(model, val_loader)
    trainer.observe(metric=accuracy, cost=training_time)
```

#### TensorFlow/Keras

```python
import tensorflow as tf
from adaptive_trainer import AdaptiveTrainer

trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train)})
    model = tf.keras.Sequential([...])
    model.compile(optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=plan['max_iter'],
                        batch_size=plan['batch_size'])
    accuracy = history.history['val_accuracy'][-1]
    trainer.observe(metric=accuracy, cost=plan['max_iter'] * plan['fidelity']**2)
```

---

## 🔧 Configuration Guide

### Alpha (Cost Penalty) Selection

| Alpha (α)   | Behavior         | Accuracy Weight | Cost Weight | Use Case                  |
| ----------- | ---------------- | --------------- | ----------- | ------------------------- |
| **0.1-0.3** | Accuracy-focused | 70-90%          | 10-30%      | Production models         |
| **0.5**     | Balanced         | 50%             | 50%         | General experimentation   |
| **0.7-0.9** | Cost-focused     | 10-30%          | 70-90%      | Budget-constrained sweeps |

**Formula:** `Reward = Accuracy - (α × Cost)`

**Recommendation:** Start with α = 0.3 for most use cases.

### Fidelity Levels

| Fidelity | Training Budget   | Use Case                          | Cost (f²) |
| -------- | ----------------- | --------------------------------- | --------- |
| **0.2**  | 20% (exploration) | Early phase, cheap evaluations    | 0.04      |
| **0.5**  | 50% (mid-range)   | Balanced exploration-exploitation | 0.25      |
| **0.75** | 75% (high)        | Late phase, refinement            | 0.56      |
| **1.0**  | 100% (maximum)    | Final convergence, best configs   | 1.00      |

**Default:** `[0.5, 0.75, 1.0]` (recommended for HPO)

---

## 🔄 Reproducibility

### Experiment Configuration

| Parameter   | Standard | Quick   | High-Confidence |
| ----------- | -------- | ------- | --------------- |
| **Seeds**   | 5        | 3       | 10              |
| **Rounds**  | 50       | 30      | 100             |
| **Alpha**   | 0.3      | 0.3     | 0.3             |
| **Runtime** | ~30 sec  | ~15 sec | ~2-3 min        |

### Running Benchmarks

```bash
cd experiments

# Standard (paper results)
python final.py --mode benchmark --dataset australian --seeds 5 --rounds 50 --alpha 0.3

# Quick validation
python final.py --mode benchmark --dataset credit_g --seeds 3 --rounds 30 --alpha 0.3

# High confidence
python final.py --mode benchmark --dataset blood_transfusion --seeds 10 --rounds 100 --alpha 0.3

# NAS benchmark
python nas_benchmark.py  # 100 rounds × 10 seeds

# Convergence analysis
python convergence_analysis.py  # All datasets, convergence curves
```

---

## ❓ FAQ

### Why is Hagfish faster than DEHB/SMAC3?

Hagfish uses **adaptive budget allocation** with intelligent escalation and pruning:

- **Early phase:** High fidelity for rapid exploration
- **Mid phase:** Mixed fidelity (70:30 high:medium ratio)
- **Late phase:** Aggressive pruning when performance saturates

DEHB/SMAC3 use fixed budget schedules that waste compute on diminishing returns.

### When should I use Hagfish?

**Use Hagfish when:**

- ✅ Training cost is significant (large models, expensive evaluations)
- ✅ You need fast convergence (early stopping, limited time)
- ✅ Accuracy-cost tradeoff matters (production constraints, budgets)

**Use traditional HPO when:**

- ❌ Training is extremely fast (seconds per trial)
- ❌ You only care about peak accuracy (infinite budget)
- ❌ Single-fidelity optimization (no budget levels available)

### How do I choose alpha?

**Rule of thumb:**

- Production systems: α = 0.1-0.3 (prioritize accuracy)
- Research/experimentation: α = 0.3-0.5 (balanced)
- Large-scale sweeps: α = 0.5-0.9 (prioritize cost)

**Validation:** Run quick experiment (3 seeds × 30 rounds) to test before full sweep.

### Can I use Hagfish with my framework?

**Yes!** Hagfish is framework-agnostic. It works with:

- Scikit-Learn ✅
- PyTorch ✅
- TensorFlow/Keras ✅
- JAX ✅
- XGBoost/LightGBM ✅
- Custom frameworks ✅

Just implement the `plan()` → train → `observe()` loop.

---

## 📞 Support

### Getting Help

- **GitHub Issues:** [Report bugs or request features](https://github.com/your-repo/hagfish-adaptive-trainer/issues)
- **Documentation:** You're reading it!
- **Examples:** See `experiments/` folder for working code

### Common Issues

#### ConvergenceWarning from Scikit-Learn

**Solution:** Expected during low-budget exploration. Safe to ignore or suppress:

```python
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

#### Results differ from benchmarks

**Reason:** Normal variation (±2%) due to random seeds, hardware, library versions.  
**Solution:** Use more seeds (10+) for stability. Match environment exactly for perfect reproduction.

#### Model training crashes

**Solution:** Report failure to agent:

```python
try:
    model.fit(X_train, y_train, **plan)
    accuracy = model.score(X_val, y_val)
except Exception:
    accuracy = 0.0  # Signal failure

trainer.observe(metric=accuracy, cost=0.0)  # Agent learns to avoid bad configs
```

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🔧 **State persistence** - Save/load trainer state
- 📊 **New benchmarks** - Additional datasets and tasks
- 🚀 **Parallel evaluation** - Multi-worker support
- 🧪 **Framework integrations** - More examples
- 📖 **Documentation** - Tutorials and guides

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📚 Academic Resources

### Papers & Methods

- **Hyperband:** Li et al., JMLR 2018 - Bandit-based HPO
- **BOHB:** Falkner et al., ICML 2018 - Bayesian optimization + Hyperband
- **DEHB:** Awad et al., NeurIPS 2021 - Differential evolution + Hyperband
- **SMAC3:** Lindauer et al., JMLR 2022 - Sequential model-based configuration
- **PBT:** Jaderberg et al., arXiv 2017 - Population-based training

### Citation

```bibtex
@software{hagfish2025,
  title = {Hagfish-SOTA: Adaptive Multi-Fidelity Hyperparameter Optimization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/hagfish-adaptive-trainer},
  note = {State-of-the-art convergence speed with adaptive cost efficiency}
}
```

---

## 📜 License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Last Updated:** January 20, 2026  
**Version:** 1.0.0
