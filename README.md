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

## Performance benchmarks

In comparative experiments, Hagfish Adaptive Trainer achieves competitive accuracy while using significantly fewer computational resources.

We compared **Hagfish** against industry heavyweights (**Optuna**, **Grid Search**) to measure efficiency. While Bayesian Optimization (Optuna) chases raw accuracy, **Hagfish** optimizes for the "Economic Sweet Spot."

| Strategy               | Accuracy (%) | Avg. Cost | **Reward (Efficiency)** |
| ---------------------- | -----------: | --------: | ----------------------: |
| Standard (Fixed)       |        92.69 |     1,363 |                  0.8996 |
| Random Search          |        93.45 |     1,142 |                  0.9117 |
| Grid Search            |        95.26 |     1,506 |                  0.9225 |
| Optuna (Bayesian)      |        96.90 |     4,197 |                  0.8851 |
| **Hagfish (Adaptive)** |    **93.51** |   **697** |              **0.9212** |

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Reward:** `Accuracy − (2 × 10⁻⁵ × Cost)`

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
