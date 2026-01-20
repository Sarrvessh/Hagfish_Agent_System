#!/usr/bin/env python3
"""Quick test of the bio-inspired Hagfish optimizer."""

from adaptive_trainer import AdaptiveTrainer

# Test initialization
print("=" * 60)
print("Testing Bio-inspired Hagfish Optimizer")
print("=" * 60)

trainer = AdaptiveTrainer(alpha=1e-4)
print("✓ Bio-inspired Hagfish AdaptiveTrainer loaded")

# Test first planning (initialization)
plan1 = trainer.plan({"dataset_size": 100})
print(f"✓ First plan: pop_size={plan1['pop_size']}, max_iter={plan1['max_iter']}")

# Simulate some observations
print("\n--- Simulating training episodes ---")
for ep in range(1, 6):
    metric = 0.70 + (ep * 0.02)  # Improving metric
    cost = 500 + (ep * 50)
    trainer.observe(metric=metric, cost=cost)
    plan = trainer.plan({"dataset_size": 100})
    print(f"Ep {ep}: metric={metric:.2f}, cost={cost} → pop_size={plan['pop_size']}, max_iter={plan['max_iter']}")

# Test stagnation and slime burst
print("\n--- Testing Slime Mechanics (Stagnation) ---")
for ep in range(6, 11):
    metric = 0.80  # Stagnation - no improvement
    cost = 1000
    trainer.observe(metric=metric, cost=cost)
    plan = trainer.plan({"dataset_size": 100})
    stag = trainer.memory.stagnation_count
    slime = trainer.memory.slime_intensity
    print(f"Ep {ep}: metric={metric:.2f}, stagnation={stag}, slime_intensity={slime:.3f} → pop_size={plan['pop_size']}, max_iter={plan['max_iter']}")

print("\n✓ All tests passed!")
print(f"✓ Elite path: {trainer.memory.elite_path}")
print(f"✓ Slime trails: {len(trainer.memory.slime_trails)} marked configs")
print(f"✓ Final slime intensity: {trainer.memory.slime_intensity:.3f}")
