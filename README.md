# hagfish-adaptive-trainer

Adaptive Training Budget Optimization for supervised ML models using a bandit-based agentic framework.

This project provides a compact framework for experimenting with budget-aware training strategies (batch sizes, epochs, reserved capacity) driven by a simple planner/critic/memory loop and a small ML workload runner.

---

## Quick summary

- Package name: `hagfish-adaptive-trainer`
- Correct import path: `from adaptive_trainer import AdaptiveTrainer`
- Version: `0.1.1`
- License: MIT

---

## Features

- PlannerAgent: a lightweight rule-based planner that proposes training budgets
- CriticAgent: assesses outcomes and determines whether a new allocation is better
- AgentMemory: stores episode history and best-known result for bandit-style analysis
- AgenticLoop: an end-to-end loop that runs training jobs and updates memory (useful for experiments)
- AdaptiveTrainer: a compact external-facing wrapper for planning and observing outcomes

---

## Installation

Install from PyPI :

pip install hagfish-adaptive-trainer

Or install from source:

pip install -e .

---

## Basic usage examples

Python example (recommended):

```python
from adaptive_trainer import AdaptiveTrainer

# Create an adapter
trainer = AdaptiveTrainer(alpha=1e-4)

# Ask for a training budget given a context
budget = trainer.plan({"dataset_size": 100})
print("Planner proposed:", budget)

# After running your training job externally, report back the observed metric and cost
trainer.observe(metric=0.85, cost=1000.0, params=budget, episode=1)
```

Agentic loop example (runs small internal solver for experiments):

```python
from adaptive_trainer.optimizer import AgenticLoop
import numpy as np

D = np.zeros((40, 40))  # placeholder distance matrix for compatibility with SolverAgent
loop = AgenticLoop(D)
results = loop.run(episodes=3, verbose=True)
print(results["best_distance"])
```

---

## API reference (high level)

- `AdaptiveTrainer(alpha: float = 1e-4)`

  - `plan(context: dict) -> dict` : returns a budget proposal (keys: `pop_size`, `max_iter`, `elite_size`)
  - `observe(metric: float, cost: float, params: dict = None, episode: int = None, elapsed_time: float = 0.0)` : record results

- `AgenticLoop(dist_matrix: np.ndarray)`
  - `run(episodes: int = 5, base_seed: int = 42, verbose: bool = True)` : run experiments end-to-end

Refer to the module-level docstrings in `adaptive_trainer` for in-depth details on PlannerAgent, CriticAgent, and AgentMemory.

---

## Warnings & behavior notes

- You can suppress warnings globally: the package intentionally does not disable warnings across the Python process.

- Expected convergence warnings: some scikit-learn solvers (used internally for the toy workload) may emit `ConvergenceWarning` for certain configurations (for example, if a solver is run with too few iterations for the dataset/problem). The project uses `SGDClassifier` with `partial_fit` in the internal solver to avoid persistent global ConvergenceWarnings; however, if you plug in different estimators or use different solver settings you may see convergence-related warnings from scikit-learn. These are informative and expected in some experimental settings.

Recommendations:

- If you want to silence these warnings locally, use Python's `warnings` module with narrow scope and restore filters afterwards; e.g.:

```python
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # run code that triggers the warning here
```

- Avoid global suppression such as `warnings.filterwarnings("ignore")` at module import time since that hides important diagnostics for other packages and users.

---

## Backwards compatibility and stability policy 🔒

- This repository preserves the existing public APIs (e.g., `AdaptiveTrainer.plan`, `AdaptiveTrainer.observe`, `AgenticLoop.run`) to remain backward compatible with users of version `0.1.0`.

- When upgrading please check the changelog (below) for non-breaking changes.

---

## Changelog

- v0.1.1 — Documentation, metadata and packaging updates (no logic changes). Bumped package metadata and README.
- v0.1.0 — Initial release.

---

## Contributing & reporting issues

Contributions are welcome. Please open a GitHub issue describing the bug or feature request. For pull requests, maintain the project's testing style and run the test suite before submitting.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
