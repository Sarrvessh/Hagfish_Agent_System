# API Reference

Complete API documentation for Hagfish-SOTA.

---

## Core Classes

### `AdaptiveTrainer`

Main class for adaptive hyperparameter optimization with multi-fidelity budget allocation.

#### Constructor

```python
AdaptiveTrainer(alpha: float = 2e-5)
```

**Parameters:**

- `alpha` (float, optional): Cost penalty coefficient. Controls accuracy vs cost trade-off.
  - Default: `2e-5`
  - Range: `1e-6` (accuracy-focused) to `1e-3` (cost-focused)
  - Recommended: `1e-5` for balanced, `0.3` for HPOBench benchmarks
  - Formula: `Reward = Accuracy - (α × Cost)`

**Returns:**

- `AdaptiveTrainer` instance

**Example:**

```python
# Accuracy-focused (production)
trainer = AdaptiveTrainer(alpha=1e-6)

# Balanced (recommended)
trainer = AdaptiveTrainer(alpha=1e-5)

# Cost-focused (large sweeps)
trainer = AdaptiveTrainer(alpha=1e-4)

# HPOBench benchmarks
trainer = AdaptiveTrainer(alpha=0.3)
```

---

#### Methods

### `plan(context: Dict[str, Any]) -> Dict[str, Any]`

Request a training budget based on historical performance and current context.

**Parameters:**

- `context` (dict): Context information for planning
  - **Required:**
    - `dataset_size` (int): Number of training samples
  - **Optional:**
    - `episode_num` (int): Current episode number (0-indexed)
    - `progress_ratio` (float): Completion percentage (0.0-1.0)
    - `metric_history` (list): Past metric values
    - Any custom context fields

**Returns:**

- `dict`: Training plan with budget recommendations
  - `fidelity` (float): Training intensity level (0.2-1.0)
  - `batch_size` (int): Recommended batch size
  - `max_iter` (int): Number of epochs/iterations
  - `pop_size` (int): Population size (for evolutionary algorithms)
  - `elite_size` (int): Elite count (for evolutionary algorithms)

**Raises:**

- `ValueError`: If `dataset_size` is missing or invalid
- `TypeError`: If context is not a dictionary

**Example:**

```python
# Minimal context
plan = trainer.plan({"dataset_size": 1000})

# Full context
plan = trainer.plan({
    "dataset_size": 1000,
    "episode_num": 5,
    "progress_ratio": 0.1
})

# Use returned plan
model = MLPClassifier(
    max_iter=plan['max_iter'],
    batch_size=plan['batch_size']
)
```

**Fidelity Interpretation:**

| Fidelity | Meaning                | Batch Size | Epochs  | Data Fraction |
| -------- | ---------------------- | ---------- | ------- | ------------- |
| 0.2      | Very low (exploration) | 8-16       | 10-20   | 10-20%        |
| 0.5      | Low-medium             | 16-32      | 50-75   | 50%           |
| 0.75     | High                   | 32-64      | 75-150  | 75%           |
| 1.0      | Maximum (exploitation) | 64-128     | 100-200 | 100%          |

---

### `observe(metric: float, cost: float, **kwargs) -> None`

Report training results back to the agent for learning and adaptation.

**Parameters:**

- `metric` (float): Model performance metric
  - Range: typically 0.0-1.0 (accuracy, F1, AUC, etc.)
  - Higher is better
  - Can be any float (loss values should be negated)
- `cost` (float): Computational cost incurred
  - Typically calculated as: `max_iter × fidelity²`
  - Or actual wall-clock time
  - Higher means more expensive
- `**kwargs` (optional): Additional context
  - `params` (dict): Training parameters used
  - `timestamp` (float): Training time
  - `converged` (bool): Whether training converged
  - `accuracy` (float): Raw accuracy (if metric is reward)
  - Any custom fields

**Returns:**

- None

**Side Effects:**

- Updates internal memory and reward history
- Influences future `plan()` calls
- May trigger escalation or pruning decisions

**Example:**

```python
# Basic usage
accuracy = model.score(X_val, y_val)
cost = plan['max_iter'] * plan['fidelity']**2
trainer.observe(metric=accuracy, cost=cost)

# With additional context
trainer.observe(
    metric=accuracy,
    cost=cost,
    params=plan,
    timestamp=time.time(),
    converged=True
)

# Failure handling
try:
    model.fit(X_train, y_train, **plan)
    accuracy = model.score(X_val, y_val)
except Exception as e:
    accuracy = 0.0  # Signal failure

trainer.observe(metric=accuracy, cost=0.0)
```

**Cost Calculation Examples:**

```python
# Quadratic cost model (standard)
cost = max_iter * fidelity**2

# Linear cost model
cost = max_iter * fidelity

# Wall-clock time
import time
start = time.time()
model.fit(X_train, y_train)
cost = time.time() - start

# Custom cost (FLOPs, GPU time, etc.)
cost = calculate_flops(model) * fidelity
```

---

## Internal Components

These classes are used internally by `AdaptiveTrainer`. You typically don't need to interact with them directly, but they're documented for advanced use cases.

### `PlannerAgent`

Proposes training budgets based on historical performance.

**Methods:**

- `plan(ep: int) -> Dict`: Generate training plan for episode `ep`

**Internal State:**

- `stagnation_count`: Number of consecutive stagnations
- `reward_trend`: Recent reward moving average
- `best_config`: Best configuration so far

### `CriticAgent`

Evaluates outcomes and classifies performance changes.

**Methods:**

- `evaluate(reward: float, prev_reward: float) -> str`: Classify outcome
  - Returns: `"improvement"`, `"stagnation"`, or `"saturation"`

**Thresholds:**

- Improvement: `reward > prev_reward + ε`
- Stagnation: `|reward - prev_reward| < ε`
- Saturation: `reward < prev_reward - ε` and `reward_trend` flat

### `AgentMemory`

Tracks historical performance and detects patterns.

**Methods:**

- `update(reward: float) -> None`: Add new reward to history
- `is_saturated() -> bool`: Check if performance has plateaued
- `get_trend() -> float`: Get recent reward trend

**Internal State:**

- `reward_history`: List of past rewards
- `window_size`: Moving average window (default: 5)

---

## Multi-Fidelity Policies

### `HagfishPolicy` (v3)

Default multi-fidelity policy used by `AdaptiveTrainer`.

**Features:**

- Three fidelity levels: [0.5, 0.75, 1.0]
- Adaptive selection based on progress
- Phase-based strategy:
  - Early (0-50%): High fidelity (1.0) with 70% probability
  - Mid (50-70%): Mixed fidelity (70:30 high:medium)
  - Late (70-85%): Weighted selection
  - Saturation: Aggressive pruning

**Configuration:**

```python
trainer = AdaptiveTrainer(alpha=0.3)
# Uses HagfishPolicy automatically
```

### Other Policies

Available in `bandit_policies.py` for comparison:

- `FixedPolicy`: Always uses fidelity = 1.0
- `RandomPolicy`: Random fidelity selection
- `CheapGreedyPolicy`: Prefers low fidelity (0.2-0.5)
- `EpsilonGreedyPolicy`: ε-greedy with exploration
- `SuccessiveHalvingPolicy`: Hyperband-style halving
- `HyperbandPolicy`: Full Hyperband algorithm
- `PBTPolicy`: Population Based Training
- `OptunaPolicy`: TPE-based Bayesian optimization

---

## Constants & Defaults

### Fidelity Levels

```python
DEFAULT_FIDELITIES = [0.5, 0.75, 1.0]  # HagfishPolicy v3
```

### Cost Model

```python
def compute_cost(fidelity: float) -> float:
    """Quadratic cost scaling"""
    return 0.04 * fidelity**2
```

### Alpha Recommendations

```python
ALPHA_PRODUCTION = 1e-6    # Maximum accuracy
ALPHA_BALANCED = 1e-5      # General use
ALPHA_COST_FOCUSED = 1e-4  # Large-scale sweeps
ALPHA_HPOBENCH = 0.3       # Benchmark experiments
```

---

## Error Handling

### Common Exceptions

**`ValueError`:**

- Missing required context keys
- Invalid parameter values
- Negative costs or metrics

**`TypeError`:**

- Wrong type for context (not dict)
- Wrong type for metric/cost (not float)

**Example Handling:**

```python
try:
    plan = trainer.plan({"dataset_size": len(X_train)})
except ValueError as e:
    print(f"Invalid context: {e}")
    plan = {"fidelity": 1.0, "max_iter": 100}  # Fallback

try:
    trainer.observe(metric=accuracy, cost=cost)
except TypeError as e:
    print(f"Invalid observation: {e}")
```

---

## Advanced Usage

### Custom Cost Functions

```python
def custom_cost(plan: Dict, actual_time: float) -> float:
    """Mix of planned and actual cost"""
    planned_cost = plan['max_iter'] * plan['fidelity']**2
    return 0.5 * planned_cost + 0.5 * actual_time

trainer.observe(metric=accuracy, cost=custom_cost(plan, train_time))
```

### Multi-Objective Rewards

```python
def multi_objective_reward(accuracy: float, f1: float, cost: float) -> float:
    """Combine multiple metrics"""
    return 0.7 * accuracy + 0.3 * f1 - 0.3 * cost

trainer.observe(
    metric=multi_objective_reward(acc, f1, cost),
    cost=cost,
    accuracy=acc,  # Track separately
    f1=f1
)
```

### State Inspection

```python
# Access internal state (advanced)
print(f"Stagnation count: {trainer.planner.stagnation_count}")
print(f"Best reward: {trainer.memory.reward_history[-1] if trainer.memory.reward_history else 0}")
print(f"Saturation detected: {trainer.memory.is_saturated()}")
```

---

## Type Hints

Complete type signatures for type checking:

```python
from typing import Dict, Any, Optional, List

class AdaptiveTrainer:
    def __init__(self, alpha: float = 2e-5) -> None: ...

    def plan(self, context: Dict[str, Any]) -> Dict[str, Any]: ...

    def observe(
        self,
        metric: float,
        cost: float,
        **kwargs: Any
    ) -> None: ...
```

---

## Deprecations & Changes

### Version 1.0.0 (Current)

- **New:** HagfishPolicy v3 (three fidelity levels)
- **Changed:** Default alpha from `2e-5` to `0.3` for benchmarks
- **Deprecated:** Old two-level fidelity system

### Migration from 0.x

```python
# Old (0.x)
trainer = AdaptiveTrainer(alpha=2e-5)
plan = trainer.plan(ep=5, dataset_size=1000)

# New (1.0)
trainer = AdaptiveTrainer(alpha=0.3)  # Or keep 2e-5 for production
plan = trainer.plan({"episode_num": 5, "dataset_size": 1000})
```

---

## Performance Tips

### Memory Efficiency

```python
# Clear history periodically for long runs
if episode % 100 == 0:
    trainer.memory.reward_history = trainer.memory.reward_history[-20:]  # Keep last 20
```

### Batch Processing

```python
# Evaluate multiple configs in parallel
plans = [trainer.plan({"dataset_size": len(X)}) for _ in range(10)]

# Train in parallel (pseudo-code)
results = parallel_train(plans)

# Report results
for plan, (accuracy, cost) in zip(plans, results):
    trainer.observe(metric=accuracy, cost=cost, params=plan)
```

### Warm Start

```python
# Initialize with known good configs
for _ in range(5):
    plan = trainer.plan({"dataset_size": len(X)})
    # Use high fidelity initially
    plan['fidelity'] = 1.0
    trainer.observe(metric=0.8, cost=2.0)  # Seed with baseline

# Now run adaptive optimization
for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X)})
    # ... normal training ...
```

---

## Testing

### Unit Tests

```python
def test_plan_basic():
    trainer = AdaptiveTrainer(alpha=0.3)
    plan = trainer.plan({"dataset_size": 1000})

    assert 'fidelity' in plan
    assert 0.2 <= plan['fidelity'] <= 1.0
    assert plan['max_iter'] > 0

def test_observe_updates():
    trainer = AdaptiveTrainer(alpha=0.3)
    trainer.plan({"dataset_size": 1000})

    initial_count = len(trainer.memory.reward_history)
    trainer.observe(metric=0.85, cost=1.5)

    assert len(trainer.memory.reward_history) == initial_count + 1
```

---

## See Also

- **[Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- **[Documentation Index](INDEX.md)** - Full documentation
- **[Benchmark Results](../experiments/comprehensive_benchmark_results.md)** - Performance analysis

---

**Questions?** [Open an issue on GitHub](https://github.com/your-repo/hagfish-adaptive-trainer/issues)
