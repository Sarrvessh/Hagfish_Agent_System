"""
═══════════════════════════════════════════════════════════════════════════════
CONVERGENCE ANALYSIS FOR HAGFISH-SOTA (ISSUE #8)
═══════════════════════════════════════════════════════════════════════════════

Generates comprehensive convergence evidence to support the claim:
"Hagfish reaches 95% of max accuracy in 6.4 episodes (on average)"

OUTPUTS:
1. 2×4 convergence curve grid (one per dataset)
2. Summary table: Method | Avg Episodes to 95% | Std Dev
3. Statistical comparison table
4. Detailed convergence metrics CSV

USAGE:
    python convergence_analysis.py --seeds 10 --rounds 50 --alpha 0.3

DATASETS:
    All 8 HPOBench datasets used in the paper

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Import benchmark infrastructure
try:
    from hpo_benchmarks import HPOBench
    SIMPLE_HPO_AVAILABLE = True
except ImportError:
    SIMPLE_HPO_AVAILABLE = False
    print("⚠️  simple-hpo-bench not installed. Run: pip install simple-hpo-bench")
    exit(1)

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

from adaptive_trainer import AdaptiveTrainer
from final import (
    HPOEnv, BasePolicy, FixedPolicy, RandomPolicy, CheapGreedyPolicy,
    EpsilonGreedyPolicy, SuccessiveHalvingPolicy, HyperbandPolicy,
    PBTPolicy, OptunaPolicy, HagfishSOTAPolicy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONVERGENCE METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_episodes_to_threshold(
    accuracies: List[float],
    threshold_pct: float = 0.95
) -> int:
    """
    Calculate episode at which method reaches threshold% of final best accuracy.
    
    Parameters
    ----------
    accuracies : List[float]
        Accuracy trajectory over episodes
    threshold_pct : float
        Percentage of max accuracy to reach (default: 0.95 for 95%)
        
    Returns
    -------
    int
        Episode number (1-indexed) when threshold is first reached
        Returns len(accuracies) if threshold never reached
    """
    if not accuracies:
        return 0
    
    max_acc = max(accuracies)
    target = max_acc * threshold_pct
    
    for i, acc in enumerate(accuracies):
        if acc >= target:
            return i + 1  # 1-indexed episode number
    
    return len(accuracies)  # Never reached threshold


def compute_normalized_convergence_curve(
    accuracies: List[float]
) -> List[float]:
    """
    Normalize accuracy trajectory to % of final best accuracy.
    
    Returns values in [0, 1] where 1.0 = final best accuracy.
    """
    if not accuracies:
        return []
    
    max_acc = max(accuracies)
    if max_acc == 0:
        return [0.0] * len(accuracies)
    
    return [acc / max_acc for acc in accuracies]


def compute_auc_normalized(normalized_curve: List[float]) -> float:
    """
    Area under normalized convergence curve (efficiency metric).
    
    Higher AUC = faster convergence to maximum.
    Perfect score: 1.0 (reaches 100% at episode 1 and stays there)
    """
    if not normalized_curve:
        return 0.0
    
    return float(np.trapz(normalized_curve, dx=1.0)) / len(normalized_curve)


def compute_convergence_speed(
    accuracies: List[float],
    window: int = 3
) -> float:
    """
    Average improvement per episode (early episodes).
    
    Parameters
    ----------
    accuracies : List[float]
        Accuracy trajectory
    window : int
        Number of initial episodes to measure
        
    Returns
    -------
    float
        Average accuracy gain per episode
    """
    if len(accuracies) < 2:
        return 0.0
    
    end_idx = min(window, len(accuracies))
    total_gain = accuracies[end_idx - 1] - accuracies[0]
    
    return total_gain / end_idx


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_single_policy(
    policy: BasePolicy,
    env: HPOEnv,
    num_rounds: int,
    alpha: float
) -> Dict:
    """Run single policy and return full trajectory."""
    accuracies = []
    costs = []
    rewards = []
    total_cost = 0.0
    
    for ep in range(1, num_rounds + 1):
        if isinstance(policy, HagfishSOTAPolicy):
            plan = policy.plan(ep, total_episodes=num_rounds)
        else:
            plan = policy.plan(ep)
        
        fidelity = plan.get("fidelity", 1.0)
        result = env.evaluate(fidelity=fidelity)
        
        accuracy = result.accuracy
        cost = result.cost
        reward = accuracy - (alpha * cost)
        
        policy.observe(
            accuracy=accuracy,
            cost=cost,
            reward=reward,
            fidelity=fidelity,
            episode=ep
        )
        
        total_cost += cost
        accuracies.append(accuracy)
        costs.append(cost)
        rewards.append(reward)
    
    # Compute convergence metrics
    conv_95 = compute_episodes_to_threshold(accuracies, 0.95)
    conv_90 = compute_episodes_to_threshold(accuracies, 0.90)
    conv_99 = compute_episodes_to_threshold(accuracies, 0.99)
    
    normalized = compute_normalized_convergence_curve(accuracies)
    auc_norm = compute_auc_normalized(normalized)
    speed = compute_convergence_speed(accuracies, window=5)
    
    return {
        "accuracies": accuracies,
        "costs": costs,
        "rewards": rewards,
        "total_cost": total_cost,
        "best_accuracy": max(accuracies) if accuracies else 0.0,
        "final_accuracy": accuracies[-1] if accuracies else 0.0,
        "convergence_95": conv_95,
        "convergence_90": conv_90,
        "convergence_99": conv_99,
        "normalized_curve": normalized,
        "auc_normalized": auc_norm,
        "convergence_speed": speed,
    }


def run_convergence_benchmark(
    dataset_name: str,
    num_seeds: int,
    num_rounds: int,
    alpha: float
) -> Dict[str, List[Dict]]:
    """Run convergence benchmark for one dataset across all methods."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name} | Seeds: {num_seeds} | Rounds: {num_rounds}")
    logger.info(f"{'='*80}")
    
    if not SIMPLE_HPO_AVAILABLE:
        logger.error("simple-hpo-bench not installed!")
        return {}
    
    # Initialize environment
    bench = HPOBench(dataset_name)  # Positional argument, not keyword
    env = HPOEnv(bench=bench)
    
    all_results: Dict[str, List[Dict]] = {}
    
    for seed in range(num_seeds):
        logger.info(f"  Seed {seed + 1}/{num_seeds}...")
        
        random.seed(seed)
        np.random.seed(seed)
        
        # All policies
        policies = {
            "Fixed": FixedPolicy(),
            "Random": RandomPolicy(),
            "CheapGreedy": CheapGreedyPolicy(),
            "EpsilonGreedy": EpsilonGreedyPolicy(),
            "SuccessiveHalving": SuccessiveHalvingPolicy(),
            "Hyperband": HyperbandPolicy(),
            "PBT": PBTPolicy(),
            "Hagfish-SOTA": HagfishSOTAPolicy(alpha=alpha),
        }
        
        if OPTUNA_AVAILABLE:
            policies["Optuna"] = OptunaPolicy(alpha=alpha)
        
        for name, policy in policies.items():
            if isinstance(policy, HagfishSOTAPolicy):
                policy.reset()
            
            result = run_single_policy(policy, env, num_rounds, alpha)
            
            if name not in all_results:
                all_results[name] = []
            all_results[name].append(result)
    
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_convergence_grid(
    all_datasets_results: Dict[str, Dict[str, List[Dict]]],
    output_path: str = "convergence_curves_grid.png"
) -> None:
    """
    Create 2×4 grid of convergence curves (one per dataset).
    
    Each subplot shows normalized accuracy (% of max) vs episode for all methods.
    """
    datasets = list(all_datasets_results.keys())
    num_datasets = len(datasets)
    
    # Create 2×4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Color scheme
    method_colors = {
        "Hagfish-SOTA": "#e74c3c",  # Red (highlight)
        "Optuna": "#3498db",  # Blue
        "Hyperband": "#2ecc71",  # Green
        "PBT": "#f39c12",  # Orange
        "SuccessiveHalving": "#9b59b6",  # Purple
        "EpsilonGreedy": "#1abc9c",  # Teal
        "CheapGreedy": "#95a5a6",  # Gray
        "Random": "#34495e",  # Dark gray
        "Fixed": "#7f8c8d",  # Light gray
    }
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        results = all_datasets_results[dataset]
        
        for method_name in sorted(results.keys()):
            runs = results[method_name]
            
            # Extract normalized curves
            curves = [r["normalized_curve"] for r in runs]
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            
            episodes = np.arange(1, len(mean_curve) + 1)
            
            color = method_colors.get(method_name, "#95a5a6")
            linewidth = 2.5 if method_name == "Hagfish-SOTA" else 1.5
            alpha_line = 1.0 if method_name == "Hagfish-SOTA" else 0.7
            
            ax.plot(
                episodes,
                mean_curve * 100,  # Convert to percentage
                label=method_name,
                color=color,
                linewidth=linewidth,
                alpha=alpha_line,
                marker='o' if method_name == "Hagfish-SOTA" else None,
                markersize=3,
                markevery=max(1, len(episodes) // 8),
            )
            
            # Confidence band (only for Hagfish)
            if method_name == "Hagfish-SOTA":
                ax.fill_between(
                    episodes,
                    (mean_curve - std_curve) * 100,
                    (mean_curve + std_curve) * 100,
                    color=color,
                    alpha=0.15,
                )
        
        # 95% threshold line
        ax.axhline(y=95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='95% Target')
        
        # Styling
        ax.set_title(f"{dataset.replace('_', ' ').title()}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Episode", fontweight='bold', fontsize=10)
        ax.set_ylabel("% of Max Accuracy", fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        
        if idx == 0:  # Legend only on first subplot
            ax.legend(fontsize=7, loc='lower right', frameon=True, shadow=True, ncol=1)
    
    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        "Convergence Speed: % of Max Accuracy vs Episode (All Datasets)",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved convergence grid: {output_path}")
    plt.close()


def plot_convergence_summary_bars(
    summary_df: pd.DataFrame,
    output_path: str = "convergence_summary_bars.png"
) -> None:
    """Bar chart comparing average episodes to 95% across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by convergence speed (lower is better)
    summary_sorted = summary_df.sort_values("Mean_Conv_95")
    
    methods = summary_sorted["Method"]
    means = summary_sorted["Mean_Conv_95"]
    stds = summary_sorted["Std_Conv_95"]
    
    # Color Hagfish differently
    colors = ["#e74c3c" if "Hagfish" in m else "#3498db" for m in methods]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Highlight Hagfish bar
    for i, method in enumerate(methods):
        if "Hagfish" in method:
            bars[i].set_linewidth(2.5)
    
    ax.set_ylabel("Episodes to 95% of Max Accuracy", fontweight='bold', fontsize=12)
    ax.set_xlabel("Method", fontweight='bold', fontsize=12)
    ax.set_title(
        "Convergence Speed Comparison (Lower is Better)",
        fontweight='bold',
        fontsize=14
    )
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved convergence bar chart: {output_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def compute_convergence_statistics(
    all_datasets_results: Dict[str, Dict[str, List[Dict]]]
) -> pd.DataFrame:
    """
    Aggregate convergence statistics across all datasets.
    
    Returns DataFrame with columns:
    - Method
    - Mean_Conv_95 (average episodes to 95%)
    - Std_Conv_95 (standard deviation)
    - Min_Conv_95, Max_Conv_95
    - Mean_Conv_90, Mean_Conv_99 (for 90% and 99% thresholds)
    """
    stats = []
    
    for dataset, results in all_datasets_results.items():
        for method_name, runs in results.items():
            conv_95_values = [r["convergence_95"] for r in runs]
            conv_90_values = [r["convergence_90"] for r in runs]
            conv_99_values = [r["convergence_99"] for r in runs]
            auc_values = [r["auc_normalized"] for r in runs]
            speed_values = [r["convergence_speed"] for r in runs]
            
            stats.append({
                "Dataset": dataset,
                "Method": method_name,
                "Conv_95_Mean": np.mean(conv_95_values),
                "Conv_95_Std": np.std(conv_95_values),
                "Conv_95_Min": np.min(conv_95_values),
                "Conv_95_Max": np.max(conv_95_values),
                "Conv_90_Mean": np.mean(conv_90_values),
                "Conv_99_Mean": np.mean(conv_99_values),
                "AUC_Mean": np.mean(auc_values),
                "Speed_Mean": np.mean(speed_values),
            })
    
    df = pd.DataFrame(stats)
    
    # Overall summary (across all datasets)
    summary = df.groupby("Method").agg({
        "Conv_95_Mean": ["mean", "std"],
        "Conv_90_Mean": "mean",
        "Conv_99_Mean": "mean",
        "AUC_Mean": "mean",
        "Speed_Mean": "mean",
    }).reset_index()
    
    summary.columns = [
        "Method",
        "Mean_Conv_95",
        "Std_Conv_95",
        "Mean_Conv_90",
        "Mean_Conv_99",
        "Mean_AUC",
        "Mean_Speed"
    ]
    
    return summary


def perform_statistical_tests(
    all_datasets_results: Dict[str, Dict[str, List[Dict]]]
) -> pd.DataFrame:
    """
    Compare Hagfish convergence speed vs all baselines.
    
    Returns DataFrame with p-values and effect sizes.
    """
    # Aggregate Hagfish conv_95 values across all datasets
    hagfish_conv = []
    baseline_conv = {method: [] for method in all_datasets_results[list(all_datasets_results.keys())[0]].keys() if method != "Hagfish-SOTA"}
    
    for dataset, results in all_datasets_results.items():
        if "Hagfish-SOTA" in results:
            hagfish_conv.extend([r["convergence_95"] for r in results["Hagfish-SOTA"]])
        
        for method_name, runs in results.items():
            if method_name != "Hagfish-SOTA" and method_name in baseline_conv:
                baseline_conv[method_name].extend([r["convergence_95"] for r in runs])
    
    # Perform t-tests
    comparison_results = []
    
    for method_name, values in baseline_conv.items():
        if len(values) > 0 and len(hagfish_conv) > 0:
            # Two-sided t-test
            t_stat, p_value = ttest_ind(hagfish_conv, values)
            
            # Cohen's d effect size
            mean_diff = np.mean(hagfish_conv) - np.mean(values)
            pooled_std = np.sqrt((np.var(hagfish_conv, ddof=1) + np.var(values, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            
            comparison_results.append({
                "Baseline": method_name,
                "Hagfish_Mean": np.mean(hagfish_conv),
                "Baseline_Mean": np.mean(values),
                "Difference": mean_diff,
                "P_Value": p_value,
                "Cohens_D": cohens_d,
                "Significant": "Yes" if p_value < 0.05 else "No",
            })
    
    return pd.DataFrame(comparison_results).sort_values("P_Value")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Convergence Analysis for Hagfish-SOTA")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--rounds", type=int, default=50, help="Episodes per seed")
    parser.add_argument("--alpha", type=float, default=0.3, help="Cost penalty (α)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "australian",
            "blood_transfusion",
            "car",
            "credit_g",
            "segment",
            "vehicle",
            "kr_vs_kp",
            "phoneme",
        ],
        help="Datasets to benchmark"
    )
    
    args = parser.parse_args()
    
    logger.info(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║       CONVERGENCE ANALYSIS FOR HAGFISH-SOTA (ISSUE #8)           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Seeds:    {args.seeds:<50} ║
    ║  Rounds:   {args.rounds:<50} ║
    ║  Alpha:    {args.alpha:<50} ║
    ║  Datasets: {len(args.datasets):<50} ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run benchmarks for all datasets
    all_datasets_results = {}
    
    for dataset in args.datasets:
        try:
            results = run_convergence_benchmark(
                dataset_name=dataset,
                num_seeds=args.seeds,
                num_rounds=args.rounds,
                alpha=args.alpha
            )
            all_datasets_results[dataset] = results
        except Exception as e:
            logger.error(f"❌ Failed on {dataset}: {e}")
            continue
    
    if not all_datasets_results:
        logger.error("No results collected. Exiting.")
        return
    
    # Compute statistics
    logger.info("\n" + "="*80)
    logger.info("COMPUTING CONVERGENCE STATISTICS...")
    logger.info("="*80)
    
    summary_df = compute_convergence_statistics(all_datasets_results)
    
    # Save summary table
    summary_path = output_dir / "convergence_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.2f")
    logger.info(f"\n✅ Saved summary: {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY: Episodes to 95% of Max Accuracy")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Statistical tests
    logger.info("\nPerforming statistical comparisons...")
    comparison_df = perform_statistical_tests(all_datasets_results)
    
    comparison_path = output_dir / "convergence_statistical_tests.csv"
    comparison_df.to_csv(comparison_path, index=False, float_format="%.4f")
    logger.info(f"✅ Saved statistical tests: {comparison_path}")
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS: Hagfish vs Baselines")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Generate visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS...")
    logger.info("="*80)
    
    # 2×4 convergence grid
    plot_convergence_grid(
        all_datasets_results,
        output_path=str(output_dir / "convergence_curves_grid.png")
    )
    
    # Bar chart summary
    plot_convergence_summary_bars(
        summary_df,
        output_path=str(output_dir / "convergence_summary_bars.png")
    )
    
    # Save detailed results (JSON)
    detailed_path = output_dir / "convergence_detailed_results.json"
    detailed_data = {}
    for dataset, results in all_datasets_results.items():
        detailed_data[dataset] = {}
        for method, runs in results.items():
            detailed_data[dataset][method] = {
                "convergence_95": [r["convergence_95"] for r in runs],
                "convergence_90": [r["convergence_90"] for r in runs],
                "convergence_99": [r["convergence_99"] for r in runs],
                "auc_normalized": [r["auc_normalized"] for r in runs],
            }
    
    with open(detailed_path, "w") as f:
        json.dump(detailed_data, f, indent=2)
    logger.info(f"✅ Saved detailed results: {detailed_path}")
    
    # Final summary
    hagfish_mean = summary_df[summary_df["Method"] == "Hagfish-SOTA"]["Mean_Conv_95"].values[0]
    hagfish_std = summary_df[summary_df["Method"] == "Hagfish-SOTA"]["Std_Conv_95"].values[0]
    
    logger.info(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    CONVERGENCE ANALYSIS COMPLETE                  ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Hagfish-SOTA Convergence (95% threshold):                        ║
    ║    Mean:  {hagfish_mean:.2f} episodes                             ║
    ║    Std:   {hagfish_std:.2f} episodes                              ║
    ║                                                                    ║
    ║  Outputs Generated:                                                ║
    ║    ✅ convergence_summary.csv                                     ║
    ║    ✅ convergence_statistical_tests.csv                           ║
    ║    ✅ convergence_curves_grid.png (2×4 grid)                      ║
    ║    ✅ convergence_summary_bars.png                                ║
    ║    ✅ convergence_detailed_results.json                           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
