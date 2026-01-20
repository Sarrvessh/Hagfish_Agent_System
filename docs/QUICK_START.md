# Quick Start Guide

Get up and running with Hagfish-SOTA in 5 minutes!

---

## Installation

```bash
pip install hagfish-adaptive-trainer
```

---

## Basic Example (5 Lines of Code)

```python
from adaptive_trainer import AdaptiveTrainer

trainer = AdaptiveTrainer(alpha=0.3)  # 70% accuracy, 30% cost

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train), "episode_num": episode})
    # Train your model with plan['fidelity'], plan['batch_size'], plan['max_iter']
    trainer.observe(metric=accuracy, cost=training_cost)
```

**That's it!** Hagfish adapts the budget automatically.

---

## Complete Scikit-Learn Example

```python
from adaptive_trainer import AdaptiveTrainer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Hagfish
trainer = AdaptiveTrainer(alpha=0.3)

# Run optimization loop
best_accuracy = 0.0
for episode in range(50):
    # Get training budget
    plan = trainer.plan({
        "dataset_size": len(X_train),
        "episode_num": episode
    })

    # Train model with plan
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=plan['max_iter'],
        batch_size=plan.get('batch_size', 32),
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    cost = plan['max_iter'] * plan['fidelity']**2  # Quadratic cost model

    # Report back to agent
    trainer.observe(metric=accuracy, cost=cost, params=plan)

    # Track best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"Episode {episode}: New best accuracy = {accuracy:.4f} (fidelity={plan['fidelity']:.2f})")

print(f"\nFinal best accuracy: {best_accuracy:.4f}")
```

---

## PyTorch Example

```python
import torch
import torch.nn as nn
from adaptive_trainer import AdaptiveTrainer

# Your model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize Hagfish
trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    # Get training budget
    plan = trainer.plan({"dataset_size": len(train_loader), "episode_num": episode})

    # Train model
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(plan['max_iter']):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    cost = plan['max_iter'] * plan['fidelity']**2

    # Report back
    trainer.observe(metric=accuracy, cost=cost)
```

---

## TensorFlow/Keras Example

```python
import tensorflow as tf
from adaptive_trainer import AdaptiveTrainer

# Initialize Hagfish
trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    # Get training budget
    plan = trainer.plan({"dataset_size": len(X_train), "episode_num": episode})

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train with plan
    history = model.fit(
        X_train, y_train,
        epochs=plan['max_iter'],
        batch_size=plan.get('batch_size', 32),
        validation_data=(X_val, y_val),
        verbose=0
    )

    # Get validation accuracy
    accuracy = history.history['val_accuracy'][-1]
    cost = plan['max_iter'] * plan['fidelity']**2

    # Report back
    trainer.observe(metric=accuracy, cost=cost)
```

---

## Configuration Guide

### Alpha (Cost Penalty)

Controls the accuracy vs cost tradeoff:

```python
# Accuracy-focused (production models)
trainer = AdaptiveTrainer(alpha=0.1)  # 90% accuracy, 10% cost

# Balanced (general use)
trainer = AdaptiveTrainer(alpha=0.3)  # 70% accuracy, 30% cost

# Cost-focused (large-scale sweeps)
trainer = AdaptiveTrainer(alpha=0.7)  # 30% accuracy, 70% cost
```

**Formula:** `Reward = Accuracy - (α × Cost)`

**Recommendation:** Start with α=0.3

### Context Dictionary

Provide additional context for better planning:

```python
plan = trainer.plan({
    "dataset_size": 1000,      # Required: number of samples
    "episode_num": 5,          # Optional: current episode
    "progress_ratio": 0.1,     # Optional: 0-1 completion
    "metric_history": [0.7, 0.75, 0.78]  # Optional: past metrics
})
```

### Plan Dictionary

What Hagfish returns:

```python
plan = {
    'fidelity': 0.75,        # Training intensity (0.2-1.0)
    'batch_size': 32,        # Recommended batch size
    'max_iter': 100,         # Number of epochs/iterations
    'pop_size': 32,          # Population size (for evolutionary)
    'elite_size': 2          # Elite count (for evolutionary)
}
```

Use what's relevant for your model.

---

## Running Benchmarks

### HPOBench (Single Dataset)

```bash
cd experiments

# Standard configuration (paper results)
python final.py --mode benchmark --dataset australian --seeds 5 --rounds 50 --alpha 0.3

# Quick validation
python final.py --mode benchmark --dataset credit_g --seeds 3 --rounds 30 --alpha 0.3
```

**Available datasets:**

- `australian`, `car`, `phoneme`, `vehicle`
- `kc1`, `segment`, `blood_transfusion`, `credit_g`

### Neural Architecture Search

```bash
cd experiments
python nas_benchmark.py  # ~5-10 minutes
```

### Convergence Analysis

```bash
cd experiments
python convergence_analysis.py  # Generates curves and statistics
```

---

## Understanding Results

### Key Metrics

**Accuracy:**

- Validation set performance (0-1 scale)
- Higher is better

**Cost:**

- Computational budget consumed
- Quadratic scaling: `Cost = 0.04 × fidelity²`
- Lower is better

**Pareto Frontier:**

- Non-dominated solutions (accuracy-cost tradeoff)
- Being "on frontier" means no method beats you on both metrics

**Statistical Significance:**

- p < 0.05: Significant difference
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

### Convergence Speed

**Episodes to 95% of max accuracy:**

- Hagfish-SOTA: **3.67 episodes** (2nd place)
- Fixed: 3.79 episodes (1st, but always uses max fidelity)
- Hyperband: 4.14 episodes (3rd)
- Optuna: 7.96 episodes (8th)

**Why 95%?**

- Diminishing returns beyond 95%
- Real-world stopping criterion
- Balances speed vs final accuracy

---

## Common Patterns

### Early Stopping

```python
trainer = AdaptiveTrainer(alpha=0.3)
best_accuracy = 0.0
patience = 5
no_improve = 0

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train)})
    # ... train model ...
    trainer.observe(metric=accuracy, cost=cost)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at episode {episode}")
        break
```

### Tracking History

```python
history = []

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train)})
    # ... train model ...
    trainer.observe(metric=accuracy, cost=cost)

    history.append({
        'episode': episode,
        'accuracy': accuracy,
        'cost': cost,
        'fidelity': plan['fidelity']
    })

# Analyze history
import pandas as pd
df = pd.DataFrame(history)
print(df.describe())
```

### Multi-Objective Optimization

```python
# Custom reward function
def custom_reward(accuracy, cost, alpha=0.3):
    return accuracy - alpha * cost

# Use in observe
trainer.observe(
    metric=custom_reward(accuracy, cost),
    cost=cost,
    accuracy=accuracy  # Still track raw accuracy
)
```

---

## Troubleshooting

### ConvergenceWarning from Scikit-Learn

**Cause:** Low-budget configurations during exploration  
**Solution:** Expected behavior, safe to ignore:

```python
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

### Results differ from benchmarks

**Cause:** Random seed variation, hardware differences, library versions  
**Solution:** Use more seeds (10+) for stability

**Typical variation:** ±2% is normal

### Model crashes during training

**Solution:** Report failure to agent:

```python
try:
    model.fit(X_train, y_train, **plan)
    accuracy = model.score(X_val, y_val)
except Exception as e:
    print(f"Training failed: {e}")
    accuracy = 0.0  # Signal failure

trainer.observe(metric=accuracy, cost=0.0)
```

Hagfish will learn to avoid problematic configurations.

---

## Next Steps

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Benchmark Results](../experiments/comprehensive_benchmark_results.md)** - Detailed results
- **[Documentation Index](INDEX.md)** - Full documentation

---

**Questions?** [Open an issue on GitHub](https://github.com/your-repo/hagfish-adaptive-trainer/issues)

**Ready to optimize!** 🚀
