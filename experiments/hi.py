"""
Hagfish Research Dashboard Generator
====================================

Generates a 6-panel publication-quality visualization comparing
Hagfish against Baselines and Optuna.

Outputs:
1. Learning Curves (Mean ± Std)
2. Cumulative Cost Growth
3. Reward Distribution Boxplots
4. Pareto Frontier (Cost vs Accuracy)
5. Convergence Speed (Episodes to 95% Accuracy)
6. Adaptive Behavior (Escalations vs Prunings)

Usage:
    python benchmark_dashboard.py --dataset australian --rounds 50 --seeds 10 --alpha 0.9
"""

import argparse
import logging
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# ----------------- IMPORTS -----------------
try:
    from hpo_benchmarks import HPOBench
    SIMPLE_HPO_AVAILABLE = True
except ImportError:
    SIMPLE_HPO_AVAILABLE = False
    print("⚠️  simple-hpo-bench not installed.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

from adaptive_trainer import AdaptiveTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class EvalResult:
    accuracy: float
    cost: float
    fidelity: float

# ----------------- ENVIRONMENT -----------------
class HPOEnv:
    FAILURE_REWARD: float = 0.0
    DEFAULT_OVERHEAD: float = 0.01
    NOISE_STD: float = 0.05 

    def __init__(self, bench: HPOBench, price_per_second: float = 0.2, overhead: float = DEFAULT_OVERHEAD):
        self.bench = bench
        self.price_per_second = price_per_second
        self.overhead = overhead
        self.search_space = bench.search_space
        self.metric_name = bench.metric_names[0] if bench.metric_names else 'val_acc'
        self.rng = np.random.RandomState(42)

    def sample_random_config(self) -> Dict:
        config = {}
        for param_name, values in self.search_space.items():
            config[param_name] = self.rng.choice(values)
        return config

    def evaluate(self, config: Dict, fidelity: float = 1.0) -> EvalResult:
        fidelity = float(np.clip(fidelity, 0.01, 1.0))
        try:
            result = self.bench(config)
            if not isinstance(result, dict):
                return EvalResult(accuracy=self.FAILURE_REWARD, cost=0.0, fidelity=fidelity)

            val_acc = self._extract_accuracy(result)
            if val_acc is None:
                return EvalResult(accuracy=self.FAILURE_REWARD, cost=0.0, fidelity=fidelity)

            fidelity_penalty = (1.0 - fidelity) * self.NOISE_STD
            noisy_acc = val_acc - fidelity_penalty + self.rng.normal(0, 0.005)
            noisy_acc = float(np.clip(noisy_acc, 0.0, 1.0))

            simulated_duration = (fidelity ** 2)
            cost = (simulated_duration * self.price_per_second) + (self.overhead * fidelity)

            return EvalResult(accuracy=noisy_acc, cost=float(cost), fidelity=fidelity)
        except Exception:
            return EvalResult(accuracy=self.FAILURE_REWARD, cost=0.0, fidelity=fidelity)

    def _extract_accuracy(self, result: Dict) -> Optional[float]:
        if self.metric_name in result: return float(result[self.metric_name])
        for key in ["val_acc", "accuracy", "score"]:
            if key in result: return float(result[key])
        try: return float(list(result.values())[0])
        except: return None

# ----------------- POLICIES -----------------
class BasePolicy:
    def plan(self, ep: int) -> Tuple[Optional[Dict], Dict]: raise NotImplementedError
    def observe(self, **kwargs) -> None: pass

class FixedPolicy(BasePolicy):
    def plan(self, ep: int): return None, {"fidelity": 1.0}

class RandomPolicy(BasePolicy):
    def plan(self, ep: int): return None, {"fidelity": random.choice([0.125, 0.25, 0.5, 1.0])}

class CheapGreedyPolicy(BasePolicy):
    def plan(self, ep: int): return None, {"fidelity": 0.125}

class OptunaPolicy(BasePolicy):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.study = optuna.create_study(direction="maximize")
        self.trial = None
    def plan(self, ep):
        self.trial = self.study.ask()
        fidelity = self.trial.suggest_categorical("fidelity", [0.125, 0.25, 0.5, 1.0])
        return None, {"fidelity": fidelity} 
    def observe(self, **kwargs):
        reward = kwargs.get("reward", None)
        if reward is not None and self.trial is not None:
            self.study.tell(self.trial, reward)

class HagfishPolicy(BasePolicy):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.reset()
    def reset(self):
        self.agent = AdaptiveTrainer(alpha=self.alpha)
        self.history_acc = []
    def plan(self, ep: int, total_episodes: int = 50):
        recent_std = np.std(self.history_acc[-5:]) if len(self.history_acc) > 5 else 0.1
        ctx = {"dataset_size": 100, "progress_ratio": ep / total_episodes, "recent_std": recent_std}
        p = self.agent.plan(ctx)
        
        raw_sig = p.get('pop_size', 10) 
        if raw_sig > 30: fidelity = 1.0      
        elif raw_sig > 15: fidelity = 0.5    
        elif raw_sig > 5: fidelity = 0.25    
        else: fidelity = 0.125               
        return None, {"fidelity": fidelity}
    def observe(self, **kwargs):
        metric = kwargs.get("accuracy", 0)
        cost = kwargs.get("cost", 0)
        self.history_acc.append(metric)
        self.agent.observe(metric=metric, cost=cost)

# ----------------- RUNNER & METRICS -----------------
def run_seed(env, num_rounds, alpha, seed):
    random.seed(seed)
    np.random.seed(seed)
    env.rng = np.random.RandomState(seed)
    
    policies = {
        "Fixed": FixedPolicy(),
        "Random": RandomPolicy(),
        "Cheap": CheapGreedyPolicy(),
        "Hagfish": HagfishPolicy(alpha=alpha)
    }
    if OPTUNA_AVAILABLE: policies["Optuna"] = OptunaPolicy(alpha=alpha)

    logs = []

    for name, policy in policies.items():
        if hasattr(policy, 'reset'): policy.reset()
        prev_fid = 1.0 
        
        for ep in range(1, num_rounds + 1):
            config, fid_dict = policy.plan(ep)
            if config is None: config = env.sample_random_config()
            fidelity = fid_dict.get("fidelity", 1.0)
            
            res = env.evaluate(config, fidelity)
            reward = res.accuracy - (alpha * res.cost)
            policy.observe(accuracy=res.accuracy, cost=res.cost, reward=reward)
            
            # Track Adaptive Behavior
            escalation = 1 if fidelity > prev_fid else 0
            pruning = 1 if fidelity < prev_fid else 0
            prev_fid = fidelity

            logs.append({
                "Strategy": name,
                "Seed": seed,
                "Episode": ep,
                "Accuracy": res.accuracy,
                "Cost": res.cost,
                "Fidelity": fidelity,
                "Reward": reward,
                "Escalation": escalation,
                "Pruning": pruning
            })
            
    return logs

# ----------------- ADVANCED PLOTTING -----------------
def compute_pareto_frontier(df):
    summary = df.groupby(['Strategy']).agg({'Cost': 'sum', 'Accuracy': 'mean'}).reset_index()
    summary = summary.sort_values('Cost')
    pareto_points = []
    max_acc = -1.0
    
    for _, row in summary.iterrows():
        if row['Accuracy'] > max_acc:
            pareto_points.append(row)
            max_acc = row['Accuracy']
            
    return pd.DataFrame(pareto_points)

def plot_dashboard(df: pd.DataFrame, dataset_name: str):
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle(f"HPOBench: Hagfish vs Baselines on {dataset_name}", fontsize=20, fontweight='bold', y=0.95)
    
    palette = sns.color_palette("bright")
    
    # (a) Learning Curves
    df['Smoothed_Acc'] = df.groupby(['Strategy', 'Seed'])['Accuracy'].transform(lambda x: x.rolling(5, 1).mean())
    sns.lineplot(data=df, x="Episode", y="Smoothed_Acc", hue="Strategy", ax=axes[0,0], palette=palette, linewidth=2)
    axes[0,0].set_title("(a) Learning Curves (Accuracy ± Std)", fontweight='bold')
    axes[0,0].set_ylabel("Validation Accuracy")

    # (b) Cumulative Cost
    df['Cum_Cost'] = df.groupby(['Strategy', 'Seed'])['Cost'].cumsum()
    sns.lineplot(data=df, x="Episode", y="Cum_Cost", hue="Strategy", ax=axes[0,1], palette=palette, linewidth=2)
    axes[0,1].set_title("(b) Cumulative Cost ± Std", fontweight='bold')
    axes[0,1].set_ylabel("Cumulative Cost ($)")

    # (c) Reward Distribution
    sns.boxplot(data=df, x="Strategy", y="Reward", ax=axes[1,0], palette=palette)
    axes[1,0].set_title("(c) Reward Distribution", fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)

    # (d) Pareto Frontier
    summary = df.groupby(['Strategy']).agg({'Cost': 'sum', 'Accuracy': 'mean'}).reset_index()
    sns.scatterplot(data=summary, x="Cost", y="Accuracy", hue="Strategy", s=300, ax=axes[1,1], palette=palette, zorder=5)
    
    # Draw Frontier Line
    pareto = compute_pareto_frontier(df)
    axes[1,1].plot(pareto['Cost'], pareto['Accuracy'], 'r--', linewidth=2, label='Pareto Frontier', zorder=1)
    
    for i in range(len(summary)):
        axes[1,1].text(summary.iloc[i]['Cost'], summary.iloc[i]['Accuracy'] + 0.002, 
                       summary.iloc[i]['Strategy'], fontsize=10, fontweight='bold')
        
    axes[1,1].set_title("(d) Pareto Frontier: Cost vs Accuracy", fontweight='bold')
    axes[1,1].set_xlabel("Total Cost (Lower is Better)")
    axes[1,1].set_ylabel("Mean Accuracy (Higher is Better)")
    axes[1,1].grid(True, which='major', linestyle='--', alpha=0.5)

    # (e) Convergence Speed
    # Calculate episodes to reach 95% of best accuracy per seed
    max_acc = df.groupby('Seed')['Accuracy'].max()
    conv_data = []
    for (strategy, seed), group in df.groupby(['Strategy', 'Seed']):
        target = 0.95 * max_acc[seed]
        reached = group[group['Accuracy'] >= target]
        eps = reached.iloc[0]['Episode'] if not reached.empty else 50
        conv_data.append({'Strategy': strategy, 'Episodes': eps})
    
    conv_df = pd.DataFrame(conv_data)
    sns.barplot(data=conv_df, x="Strategy", y="Episodes", ax=axes[2,0], palette=palette, capsize=.1)
    axes[2,0].set_title("(e) Convergence Speed (Fewer = Better)", fontweight='bold')
    axes[2,0].tick_params(axis='x', rotation=45)

    # (f) Adaptive Behavior
    behavior = df.groupby('Strategy')[['Escalation', 'Pruning']].mean().reset_index()
    behavior_melt = behavior.melt(id_vars="Strategy", var_name="Action", value_name="Avg Count per Episode")
    sns.barplot(data=behavior_melt, x="Strategy", y="Avg Count per Episode", hue="Action", ax=axes[2,1], palette=['#2ecc71', '#e74c3c'])
    axes[2,1].set_title("(f) Adaptive Behavior: Escalations vs Prunings", fontweight='bold')
    axes[2,1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"dashboard_{dataset_name}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Dashboard saved as: {filename}")

# ----------------- MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="credit_g")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=10) 
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    if not SIMPLE_HPO_AVAILABLE: return

    print(f"Generating Dashboard: {args.dataset} (Seeds={args.seeds}, Rounds={args.rounds})")
    try:
        bench = HPOBench(args.dataset)
        env = HPOEnv(bench, price_per_second=0.2)
    except: return

    all_logs = []
    for seed in range(args.seeds):
        logger.info(f"Seed {seed+1}/{args.seeds}...")
        logs = run_seed(env, args.rounds, args.alpha, seed)
        all_logs.extend(logs)

    df = pd.DataFrame(all_logs)
    plot_dashboard(df, args.dataset)

    # Mini Table Output
    summary = df.groupby('Strategy').agg({'Accuracy': 'mean', 'Cost': 'sum', 'Reward': 'mean'}).reset_index()
    print("\n" + "="*60)
    print(f"{'Strategy':<15} | {'Acc':<8} | {'Cost':<8} | {'Efficiency'}")
    print("-" * 60)
    for _, row in summary.iterrows():
        eff = row['Accuracy'] / (row['Cost'] + 0.001)
        print(f"{row['Strategy']:<15} | {row['Accuracy']:.4f}   | {row['Cost']:.2f}     | {eff:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()