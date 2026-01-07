"""
Hagfish-based Neural Architecture Search (NAS)
----------------------------------------------
Optimizes architecture + training budget jointly
using a reward = accuracy - alpha * cost formulation.
"""

import time
import numpy as np
from adaptive_trainer import AdaptiveTrainer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =============================
# 1. Dataset
# =============================
X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

DATASET_SIZE = len(X_train)

# =============================
# 2. Hagfish Trainer
# =============================
ALPHA = 1e-5  # Cost sensitivity
EPISODES = 20

trainer = AdaptiveTrainer(alpha=ALPHA)

best_result = {"acc": 0.0, "cost": float("inf"), "config": None}

# =============================
# 3. NAS Loop
# =============================
print("======================================================================")
print("Hagfish-based Neural Architecture Search (NAS)")
print("======================================================================")
print(f"Alpha (cost sensitivity):     {ALPHA}")
print(f"Total Episodes:               {EPISODES}")
print(f"Dataset Size:                 {DATASET_SIZE}")
print("======================================================================\n")

episode_history = []

for ep in range(1, EPISODES + 1):
    # Hagfish proposes a training budget
    plan = trainer.plan({"dataset_size": DATASET_SIZE})

    # Map budget → architecture + training config
    config = {
        "hidden_units": int(np.clip(plan["pop_size"] * 4, 32, 512)),
        "num_layers": int(np.clip(plan["elite_size"] + 1, 1, 4)),
        "epochs": int(np.clip(plan["max_iter"] // 10, 10, 100)),
        "batch_size": int(np.clip(plan["pop_size"], 16, 128)),
    }

    # Build architecture
    hidden_layers = tuple([config["hidden_units"]] * config["num_layers"])

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=config["epochs"],
        batch_size=config["batch_size"],
        random_state=42 + ep,
    )

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Evaluate
    acc = accuracy_score(y_val, model.predict(X_val))

    # Compute cost (proxy)
    cost = train_time * config["hidden_units"] * config["batch_size"]

    # Hagfish feedback
    trainer.observe(metric=acc, cost=cost, params=config)

    # Track best
    if acc > best_result["acc"]:
        best_result = {
            "acc": acc,
            "cost": cost,
            "config": config
        }

    # Store for summary
    episode_history.append({
        "episode": ep,
        "acc": acc,
        "cost": cost,
        "config": config
    })

    print(
        f"Episode {ep:02d} | "
        f"Accuracy={acc:.4f} | Cost={cost:.2f} | "
        f"Layers={config['num_layers']} | "
        f"Units={config['hidden_units']} | "
        f"Epochs={config['epochs']}"
    )

# =============================
# 4. Final Result
# =============================
print("\n" + "="*70)
print("SEARCH COMPLETE")
print("="*70)
print(f"Total Episodes Run:           {EPISODES}")
print(f"Total Evaluations:            {EPISODES}")
print("="*70)

print("\n" + "="*70)
print("BEST ARCHITECTURE FOUND")
print("="*70)
print(f"Validation Accuracy:          {best_result['acc']:.4f}")
print(f"Compute Cost:                 {best_result['cost']:.2f}")
print("\nArchitecture Configuration:")
print(f"  Number of Layers:           {best_result['config']['num_layers']}")
print(f"  Hidden Units per Layer:     {best_result['config']['hidden_units']}")
print(f"  Training Epochs:            {best_result['config']['epochs']}")
print(f"  Batch Size:                 {best_result['config']['batch_size']}")
print("="*70)
