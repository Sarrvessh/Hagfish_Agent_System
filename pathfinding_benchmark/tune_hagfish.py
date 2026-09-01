"""
Hagfish Hyperparameter Tuning for Pathfinding
==============================================
Find the best alpha and other hyperparameters for Hagfish on pathfinding tasks.
No code tampering - just proper configuration tuning.
"""

import numpy as np
import time
import sys
import os
import copy

# Add parent to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.extend([curr_dir, parent_dir])

from pathfinding_benchmark import RobotNavigation, HagfishPathfinder

def test_config(alpha, pop_size, max_iter, scenario='Trap', n_waypoints=5, n_runs=5):
    """Test a specific Hagfish configuration"""
    problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
    
    costs = []
    valid_count = 0
    
    for run in range(n_runs):
        hagfish = HagfishPathfinder(pop_size=pop_size, alpha=alpha, seed=42+run)
        
        pos, cost, _ = hagfish.optimize(
            objective_fn=problem.evaluate,
            bounds=problem.bounds,
            dim=problem.dim,
            max_iterations=max_iter
        )
        
        costs.append(cost)
        if problem.is_valid_path(pos):
            valid_count += 1
    
    return {
        'mean': np.mean(costs),
        'std': np.std(costs),
        'best': np.min(costs),
        'worst': np.max(costs),
        'valid_rate': valid_count / n_runs
    }


def grid_search():
    """Grid search over hyperparameters"""
    print("=" * 80)
    print("  HAGFISH HYPERPARAMETER TUNING - PATHFINDING")
    print("=" * 80)
    
    # Hyperparameter grid
    alphas = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    pop_sizes = [30, 40, 50]
    max_iters = [100, 150, 200]
    
    print(f"\nTesting {len(alphas)} alphas × {len(pop_sizes)} pop_sizes × {len(max_iters)} iterations")
    print(f"Total configurations: {len(alphas) * len(pop_sizes) * len(max_iters)}")
    print(f"Runs per config: 5")
    print()
    
    results = []
    
    for alpha in alphas:
        for pop_size in pop_sizes:
            for max_iter in max_iters:
                print(f"Testing: alpha={alpha:.1e}, pop={pop_size}, iter={max_iter}...", end=" ")
                
                start = time.time()
                result = test_config(alpha, pop_size, max_iter)
                elapsed = time.time() - start
                
                result['alpha'] = alpha
                result['pop_size'] = pop_size
                result['max_iter'] = max_iter
                result['time'] = elapsed
                
                results.append(result)
                
                print(f"Mean={result['mean']:.2f}, Best={result['best']:.2f}, Valid={result['valid_rate']*100:.0f}%")
    
    # Sort by best mean cost
    results.sort(key=lambda x: x['mean'])
    
    print("\n" + "=" * 80)
    print("  TOP 10 CONFIGURATIONS")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Alpha':<10} {'Pop':<6} {'Iter':<6} {'Mean':<10} {'Best':<10} {'Valid%':<8} {'Time(s)':<8}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['alpha']:<10.1e} {r['pop_size']:<6} {r['max_iter']:<6} "
              f"{r['mean']:<10.2f} {r['best']:<10.2f} {r['valid_rate']*100:<8.0f} {r['time']:<8.2f}")
    
    print("\n" + "=" * 80)
    print("  BEST CONFIGURATION")
    print("=" * 80)
    best = results[0]
    print(f"\n  Alpha:        {best['alpha']:.1e}")
    print(f"  Pop Size:     {best['pop_size']}")
    print(f"  Max Iter:     {best['max_iter']}")
    print(f"  Mean Cost:    {best['mean']:.2f}")
    print(f"  Best Cost:    {best['best']:.2f}")
    print(f"  Valid Rate:   {best['valid_rate']*100:.0f}%")
    print(f"  Time:         {best['time']:.2f}s")
    print()
    
    return results


def quick_alpha_sweep():
    """Quick alpha sweep with fixed pop_size and iterations"""
    print("=" * 80)
    print("  QUICK ALPHA SWEEP (Fixed: pop=40, iter=150)")
    print("=" * 80)
    print()
    
    alphas = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    print(f"{'Alpha':<12} | {'Mean':<10} | {'Best':<10} | {'Std':<10} | {'Valid%':<8}")
    print("-" * 60)
    
    results = []
    for alpha in alphas:
        result = test_config(alpha=alpha, pop_size=40, max_iter=150, n_runs=5)
        result['alpha'] = alpha
        results.append(result)
        
        print(f"{alpha:<12.1e} | {result['mean']:<10.2f} | {result['best']:<10.2f} | "
              f"{result['std']:<10.2f} | {result['valid_rate']*100:<8.0f}")
    
    # Find best
    best = min(results, key=lambda x: x['mean'])
    print("\n" + "=" * 60)
    print(f"✓ Best Alpha: {best['alpha']:.1e} (Mean Cost: {best['mean']:.2f})")
    print("=" * 60)
    print()
    
    return results


def compare_with_baselines():
    """Compare best Hagfish config with baselines"""
    print("=" * 80)
    print("  HAGFISH (TUNED) VS BASELINES")
    print("=" * 80)
    
    # First find best alpha
    print("\n1. Finding best alpha...")
    alpha_results = quick_alpha_sweep()
    best_alpha = min(alpha_results, key=lambda x: x['mean'])['alpha']
    
    print(f"\n2. Using best alpha: {best_alpha:.1e}")
    print("\n3. Running comparison with 10 runs...")
    
    # Compare against all available package-backed baseline optimizers
    from baselines import get_available_baseline_optimizers
    
    problem = RobotNavigation(n_waypoints=5, scenario='Trap')
    n_runs = 10
    
    algorithms = {
        f'Hagfish (alpha={best_alpha:.1e})': lambda: HagfishPathfinder(40, alpha=best_alpha),
    }
    baseline_templates = get_available_baseline_optimizers(pop_size=40)
    baseline_factories = {
        name: (lambda factory_name=name: copy.deepcopy(baseline_templates[factory_name]))
        for name in baseline_templates.keys()
    }
    algorithms.update(baseline_factories)
    
    print(f"\n{'Algorithm':<25} | {'Mean':<10} | {'Best':<10} | {'Std':<10} | {'Valid%':<8}")
    print("-" * 75)
    
    for name, algo_factory in algorithms.items():
        costs = []
        valid = 0
        
        for run in range(n_runs):
            algo = algo_factory()
            pos, cost, _ = algo.optimize(
                objective_fn=problem.evaluate,
                bounds=problem.bounds,
                dim=problem.dim,
                max_iterations=150
            )
            costs.append(cost)
            if problem.is_valid_path(pos):
                valid += 1
        
        print(f"{name:<25} | {np.mean(costs):<10.2f} | {np.min(costs):<10.2f} | "
              f"{np.std(costs):<10.2f} | {valid/n_runs*100:<8.0f}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune Hagfish hyperparameters')
    parser.add_argument('--quick', action='store_true', help='Quick alpha sweep only')
    parser.add_argument('--full', action='store_true', help='Full grid search')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines')
    
    args = parser.parse_args()
    
    if args.full:
        grid_search()
    elif args.compare:
        compare_with_baselines()
    else:
        # Default: quick alpha sweep
        quick_alpha_sweep()
