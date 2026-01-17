"""
Hagfish vs SOTA NAS: The Ultimate Benchmark (Including DARTS)
=============================================================

Scenario:
    Searching for the optimal Neural Network Architecture (CIFAR-10 proxy).
    Search Space: 5 Nodes, 5 Operations.

Competitors:
    1. Random Search (Baseline)
    2. Regularized Evolution (REA - The Evolutionary Standard)
    3. Bayesian Optimization (Optuna TPE - The Probability Standard)
    4. Successive Halving (SHA - The Pruning Standard)
    5. RL Policy Gradient (The "Old School" NAS)
    6. DARTS (Simulated Differentiable Search)
    7. Hagfish (Your Adaptive Budget Agent)

Hypothesis:
    - DARTS & Optuna will find high accuracy but have HIGH cost.
    - SHA will have LOW cost but potentially lower accuracy (too aggressive).
    - Hagfish will find the "Sweet Spot" (High Accuracy, Low Cost).
"""

import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- CONFIG ---
plt.rcParams['font.family'] = 'serif'
sns.set_style("whitegrid")

# --- IMPORTS ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Skipping Bayesian Optimization.")

try:
    from adaptive_trainer import AdaptiveTrainer
except ImportError:
    print("⚠️ 'adaptive_trainer.py' not found. Please ensure it is in the same folder.")

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT (NAS-Bench-201 Proxy)
# -----------------------------------------------------------------------------
class SyntheticNASBench:
    OPS = ["conv3x3", "conv1x1", "maxpool", "skip", "zero"]
    
    # Operation Properties (Accuracy vs Cost)
    OP_PROPS = {
        "conv3x3": {"acc": 0.15, "cost": 1.0},
        "conv1x1": {"acc": 0.12, "cost": 0.6},
        "maxpool": {"acc": 0.08, "cost": 0.3},
        "skip":    {"acc": 0.05, "cost": 0.1},
        "zero":    {"acc": 0.00, "cost": 0.0},
    }

    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes

    def random_arch(self) -> List[str]:
        return [random.choice(self.OPS) for _ in range(self.num_nodes)]

    def evaluate(self, arch: List[str], epochs: int) -> Dict:
        """Simulates training an architecture for N epochs."""
        potential = 0.40 # Base accuracy
        complexity = 0.0
        
        for op in arch:
            props = self.OP_PROPS[op]
            potential += props["acc"]
            complexity += props["cost"]
        
        # Interaction Bonus (e.g. Conv+Pool is good)
        for i in range(len(arch)-1):
            if "conv" in arch[i] and "pool" in arch[i+1]:
                potential += 0.05

        potential = np.clip(potential, 0.1, 0.96) # Max theoretical acc
        
        # Learning Curve Simulation
        # Harder models take longer to converge (higher tau)
        tau = 5.0 + (complexity * 1.5) 
        accuracy = potential * (1 - np.exp(-epochs / tau))
        
        # Stochastic Noise
        accuracy += np.random.normal(0, 0.004)
        
        # Cost Simulation (GPU Seconds)
        time_cost = (0.5 + complexity) * epochs * 0.1
        
        return {"accuracy": accuracy, "cost": time_cost}

# -----------------------------------------------------------------------------
# 2. AGENTS / ALGORITHMS
# -----------------------------------------------------------------------------

class NASAgent:
    def suggest(self, history) -> Tuple[List[str], int]:
        raise NotImplementedError
    def update(self, arch, acc, cost=0):
        pass

# --- BASELINE: RANDOM ---
class RandomSearch(NASAgent):
    def __init__(self, env, max_epochs=25):
        self.env = env
        self.max_epochs = max_epochs
    def suggest(self, history):
        return self.env.random_arch(), self.max_epochs

# --- SOTA: EVOLUTION (REA) ---
class RegularizedEvolution(NASAgent):
    def __init__(self, env, max_epochs=25, pop_size=10, sample_size=3):
        self.env = env
        self.max_epochs = max_epochs
        self.population = [] 
        self.pop_size = pop_size
        self.sample_size = sample_size
        
    def mutate(self, arch):
        new_arch = arch.copy()
        idx = random.randint(0, len(arch)-1)
        new_arch[idx] = random.choice(self.env.OPS)
        return new_arch

    def suggest(self, history):
        if len(self.population) < self.pop_size:
            return self.env.random_arch(), self.max_epochs
        sample = random.sample(self.population, self.sample_size)
        parent = max(sample, key=lambda x: x[1])[0]
        return self.mutate(parent), self.max_epochs
    
    def update(self, arch, acc, cost=0):
        self.population.append((arch, acc))
        if len(self.population) > self.pop_size:
            self.population.pop(0)

# --- SOTA: BAYESIAN OPTIMIZATION ---
class BayesianOptuna(NASAgent):
    def __init__(self, env, max_epochs=25):
        self.env = env
        self.max_epochs = max_epochs
        self.study = optuna.create_study(direction="maximize")
        self.trial = None
        
    def suggest(self, history):
        self.trial = self.study.ask()
        arch = []
        for i in range(self.env.num_nodes):
            op = self.trial.suggest_categorical(f"n{i}", self.env.OPS)
            arch.append(op)
        return arch, self.max_epochs

    def update(self, arch, acc, cost=0):
        self.study.tell(self.trial, acc)

# --- SOTA: SUCCESSIVE HALVING (ASHA) ---
class SuccessiveHalving(NASAgent):
    def __init__(self, env, max_epochs=25):
        self.env = env
        self.max_epochs = max_epochs
        self.brackets = [1, 3, 9, 25] 
        
    def suggest(self, history):
        # 70% Explore (Low Rung), 30% Promote (High Rung)
        if len(history) > 10 and random.random() < 0.3:
            candidates = [h for h in history if h['epochs'] < self.max_epochs]
            if candidates:
                best = max(candidates, key=lambda x: x['acc'])
                # Find next rung
                for e in self.brackets:
                    if e > best['epochs']:
                        return best['arch'], e
        return self.env.random_arch(), self.brackets[0]

# --- SOTA: SIMULATED DARTS (Gradient Approximation) ---
class SimulatedDARTS(NASAgent):
    """
    Simulates Differentiable Architecture Search.
    Maintains 'alpha' weights for ops and updates them via pseudo-gradient.
    Usually trains heavily (High Cost).
    """
    def __init__(self, env, max_epochs=25):
        self.env = env
        self.max_epochs = max_epochs
        # Initialize Alphas (Weights) for [Nodes, Ops]
        self.alphas = np.zeros((env.num_nodes, len(env.OPS)))
        self.lr = 0.8 # Learning rate
        self.last_indices = []

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def suggest(self, history):
        # Sample arch based on current Alphas
        probs = np.apply_along_axis(self.softmax, 1, self.alphas)
        arch = []
        self.last_indices = []
        for i in range(self.env.num_nodes):
            idx = np.random.choice(len(self.env.OPS), p=probs[i])
            arch.append(self.env.OPS[idx])
            self.last_indices.append(idx)
        
        # DARTS typically trains the Supernet fully
        return arch, self.max_epochs

    def update(self, arch, acc, cost=0):
        # Pseudo-gradient update: Move alphas towards successful ops
        baseline = 0.6 
        grad = (acc - baseline)
        for i, op_idx in enumerate(self.last_indices):
            self.alphas[i][op_idx] += self.lr * grad

# --- HAGFISH (YOURS) ---
class HagfishNAS(NASAgent):
    def __init__(self, env, max_epochs=25):
        self.env = env
        self.max_epochs = max_epochs
        # High Alpha to prioritize Cost Efficiency
        self.agent = AdaptiveTrainer(alpha=2.0) 
        self.population = [] 
        self.history_accs = []
        
    def suggest(self, history):
        # 1. Budget Decision (The Innovation)
        if len(self.history_accs) > 5:
            std = np.std(self.history_accs[-5:])
        else:
            std = 0.1
        ctx = {"dataset_size": 1000, "recent_std": std}
        plan = self.agent.plan(ctx)
        
        # Map Hagfish signal to Epochs
        signal = plan.get('pop_size', 10)
        if signal > 25: epochs = self.max_epochs          
        elif signal > 12: epochs = int(self.max_epochs * 0.6) 
        else: epochs = int(self.max_epochs * 0.2)         
        
        # 2. Architecture Search (Evolutionary Backbone)
        if len(self.population) < 5:
            arch = self.env.random_arch()
        else:
            parent = max(self.population, key=lambda x: x[1])[0]
            arch = parent.copy()
            idx = random.randint(0, len(arch)-1)
            arch[idx] = random.choice(self.env.OPS)
            
        return arch, epochs

    def update(self, arch, acc, cost):
        self.population.append((arch, acc))
        self.history_accs.append(acc)
        # Normalize Cost for Reward
        norm_cost = cost / 6.0 
        self.agent.observe(metric=acc, cost=cost)

# -----------------------------------------------------------------------------
# 3. RUNNER & PLOTTING
# -----------------------------------------------------------------------------
def run_benchmark(rounds=100, seeds=5):
    print(f"\n🚀 NAS Benchmark Ultimate (Rounds={rounds}, Seeds={seeds})...")
    
    competitors = {
        "Random": RandomSearch,
        "Evolution (REA)": RegularizedEvolution,
        "SHA (Hyperband)": SuccessiveHalving,
        "DARTS (Sim)": SimulatedDARTS,
        "Hagfish": HagfishNAS
    }
    if OPTUNA_AVAILABLE:
        competitors["Optuna (TPE)"] = BayesianOptuna

    results = {name: {"acc": [], "cost": []} for name in competitors}
    
    for seed in range(seeds):
        print(f"  Seed {seed+1}/{seeds}...")
        random.seed(seed)
        np.random.seed(seed)
        env = SyntheticNASBench()
        
        agents = {name: cls(env) for name, cls in competitors.items()}
        
        for name, agent in agents.items():
            cumulative_cost = 0
            best_acc = 0
            history = []
            
            for _ in range(rounds):
                arch, epochs = agent.suggest(history)
                res = env.evaluate(arch, epochs)
                
                history.append({'arch': arch, 'epochs': epochs, 'acc': res['accuracy']})
                
                # Update Agents
                if name == "Hagfish":
                    agent.update(arch, res['accuracy'], res['cost'])
                else:
                    agent.update(arch, res['accuracy'])
                
                cumulative_cost += res['cost']
                best_acc = max(best_acc, res['accuracy'])
                
            results[name]['acc'].append(best_acc)
            results[name]['cost'].append(cumulative_cost)

    return results

def plot_results(results):
    names = list(results.keys())
    accs = [np.mean(results[n]['acc']) for n in names]
    costs = [np.mean(results[n]['cost']) for n in names]
    
    print("\n" + "="*80)
    print(f"{'Strategy':<20} | {'Best Acc':<10} | {'Total Cost':<10} | {'Efficiency'}")
    print("-" * 80)
    for n, a, c in zip(names, accs, costs):
        eff = (a / c) * 100 if c > 0 else 0
        print(f"{n:<20} | {a:.4f}     | {c:.2f}       | {eff:.2f}")
    print("="*80)

    # Visualization
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(names))
    width = 0.35
    
    # Plot Accuracy
    bars1 = ax1.bar(x - width/2, accs, width, label='Accuracy (Higher is Better)', color='#3498db', alpha=0.9, edgecolor='black')
    ax1.set_ylabel('Best Accuracy', fontweight='bold', fontsize=12, color='#2980b9')
    ax1.set_ylim(0.85, 0.98) # Zoom in on top accuracy
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    
    # Plot Cost
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, costs, width, label='Cost (Lower is Better)', color='#e74c3c', alpha=0.9, edgecolor='black')
    ax2.set_ylabel('Compute Cost (GPU Units)', fontweight='bold', fontsize=12, color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight='bold', rotation=15)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title("NAS Benchmark: Hagfish vs 5 SOTA Algorithms", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("nas_ultimate_result.png", dpi=300)
    print("\n✅ Benchmark Graph saved to: nas_ultimate_result.png")

if __name__ == "__main__":
    results = run_benchmark(rounds=100, seeds=10)
    plot_results(results)