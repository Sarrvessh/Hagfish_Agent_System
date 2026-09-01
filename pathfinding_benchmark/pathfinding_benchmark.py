"""
Pathfinding Benchmark Suite
===========================
Comprehensive benchmark for comparing metaheuristic algorithms on 
continuous robot navigation / pathfinding problems.

Tests algorithms across:
- Multiple scenarios (Sparse, Trap, Maze, Clutter, Forest)
- Multiple waypoint counts
- Multiple independent runs for statistical significance
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PATHFINDING_PLOTS_DIR = EXPERIMENTS_DIR / 'output' / 'plots' / 'pathfinding'
PATHFINDING_LOGS_DIR = EXPERIMENTS_DIR / 'output' / 'logs' / 'pathfinding'

# Ensure repository-local packages are importable from any working directory.
repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

# Try importing the research algorithms
try:
    from adaptive_trainer import AdaptiveTrainer
    try:
        from pathfinding_benchmark.baselines import get_available_baseline_optimizers
    except ModuleNotFoundError:
        from baselines import get_available_baseline_optimizers
except ImportError as e:
    print(f"Error importing research algorithms: {e}")
    sys.exit(1)


class HagfishPathfinder:
    """
    Hagfish Adaptive Trainer wrapper for pathfinding optimization.
    Uses adaptive budgeting to guide evolutionary search for optimal waypoint placement.
    """
    def __init__(self, pop_size=40, alpha=5e-4, seed=None):
        self.pop_size = pop_size
        self.trainer = AdaptiveTrainer(alpha=alpha)
        self.best_pos = None
        self.best_cost = float('inf')
        self.rng = np.random.RandomState(seed)  # Use local RNG instead of global

    def optimize(self, objective_fn, bounds, dim, max_iterations=150):
        """Optimize pathfinding objective (minimize path cost)."""
        lower, upper = bounds
        lower = np.full(dim, lower) if np.isscalar(lower) else np.array(lower)
        upper = np.full(dim, upper) if np.isscalar(upper) else np.array(upper)

        # Initialize population
        pop = self.rng.uniform(lower, upper, (self.pop_size, dim))
        fitness = np.array([objective_fn(x) for x in pop])

        history = []

        for iteration in range(max_iterations):
            # Get adaptive plan from Hagfish
            plan = self.trainer.plan({
                "problem_size": int(dim),
                "dataset_size": int(dim),
                "iteration": int(iteration),
                "current_pop_size": int(self.pop_size),
                "max_iterations": int(max_iterations),
            })

            planned_pop_size = int(np.clip(plan.get("pop_size", self.pop_size), 16, 200))
            planned_max_iter = int(np.clip(plan.get("max_iter", max_iterations), 50, 1000))
            elite_size = int(np.clip(plan.get("elite_size", 5), 1, self.pop_size // 2))

            # Select elites (best solutions)
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = pop[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Adaptive mutation strength - more aggressive early, refine later
            progress = iteration / max_iterations
            base_sigma = 0.15 * (1 - 0.7 * progress)  # Decrease over time
            elite_factor = (elite_size / max(1, self.pop_size))
            # Let the planner's suggested budget influence exploration scale.
            budget_factor = (planned_pop_size / max(1, self.pop_size))
            iter_factor = (planned_max_iter / max(1, max_iterations))
            sigma = base_sigma * (0.5 + elite_factor) * np.clip(0.5 * budget_factor + 0.5 * iter_factor, 0.5, 1.8)

            # Generate offspring via mutation with crossover
            offspring = []
            for _ in range(self.pop_size - elite_size):
                if self.rng.random() < 0.7:  # 70% mutation
                    parent = elites[self.rng.randint(len(elites))]
                    child = parent + self.rng.normal(0, sigma * (upper - lower), dim)
                else:  # 30% crossover
                    p1 = elites[self.rng.randint(len(elites))]
                    p2 = elites[self.rng.randint(len(elites))]
                    mask = self.rng.random(dim) < 0.5
                    child = np.where(mask, p1, p2)
                    child = child + self.rng.normal(0, sigma * 0.5 * (upper - lower), dim)
                child = np.clip(child, lower, upper)
                offspring.append(child)

            offspring = np.array(offspring)
            offspring_fitness = np.array([objective_fn(x) for x in offspring])

            # Combine and select next generation
            all_pop = np.vstack([elites, offspring])
            all_fitness = np.concatenate([elite_fitness, offspring_fitness])
            select_indices = np.argsort(all_fitness)[:self.pop_size]
            pop = all_pop[select_indices]
            fitness = all_fitness[select_indices]

            # Update best
            if fitness[0] < self.best_cost:
                self.best_cost = fitness[0]
                self.best_pos = pop[0].copy()

            # Feedback to Hagfish (higher metric is better)
            # Convert cost to reward: lower cost = higher reward
            metric = 1.0 / (1.0 + self.best_cost)
            # Provide full budget params so slime trails and elite-path updates are meaningful.
            self.trainer.observe(
                metric=metric,
                cost=self.best_cost,
                params={
                    "iteration": int(iteration),
                    "pop_size": int(planned_pop_size),
                    "max_iter": int(planned_max_iter),
                    "elite_size": int(elite_size),
                },
            )

            history.append(self.best_cost)

        return self.best_pos, self.best_cost, history

# ============================================
# PATHFINDING PROBLEM CLASS
# ============================================

class RobotNavigation:
    """
    Continuous Pathfinding Benchmark
    Goal: Find k waypoints to get from Start to End while avoiding obstacles.
    
    The search space is continuous - algorithms optimize (x,y) coordinates
    for each waypoint to minimize path length while avoiding collisions.
    """
    
    def __init__(self, n_waypoints=5, scenario='Default'):
        self.n_waypoints = n_waypoints
        self.dim = n_waypoints * 2  # x, y for each waypoint
        self.bounds = (0, 100)      # Map size 100x100
        self.scenario_name = scenario
        
        # Default positions
        self.start_pos = np.array([5, 5])
        self.end_pos = np.array([95, 95])
        self.obstacles = []
        
        # Load scenario
        self.load_scenario(scenario)
        
        # Compute theoretical minimum (straight line distance)
        self.min_distance = np.linalg.norm(self.end_pos - self.start_pos)

    def load_scenario(self, name):
        """Load different standard benchmark scenarios."""
        self.scenario_name = name
        self.obstacles = []
        
        if name == 'Default':
            # Default setup with center obstacles
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            self.obstacles = [
                (50, 50, 15),  # Big center block
                (20, 80, 10),
                (80, 20, 10),
                (30, 30, 8),
                (70, 70, 8),
                (40, 60, 5),
                (60, 40, 5)
            ]
            
        elif name == 'Sparse':
            # Easy: Just a few random blocks
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            self.obstacles = [
                (50, 50, 10), (20, 70, 10), (70, 30, 10)
            ]
            
        elif name == 'Trap':
            # Medium: The classic "Bug Trap" (U-shape blocking the goal)
            self.start_pos = np.array([10, 50])
            self.end_pos = np.array([50, 50])  # Inside the trap
            # Back wall
            for y in range(30, 71, 5):
                self.obstacles.append((60, y, 4))
            # Side walls (U-shape)
            for x in range(35, 60, 5): 
                self.obstacles.append((x, 30, 4))  # Bottom
                self.obstacles.append((x, 70, 4))  # Top
                
        elif name == 'Maze':
            # Hard: S-curve maze requiring multiple direction changes
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            # First horizontal wall
            for x in range(0, 80, 5):
                self.obstacles.append((x, 33, 4))
            # Second horizontal wall (offset)
            for x in range(20, 100, 5):
                self.obstacles.append((x, 66, 4))
                
        elif name == 'Clutter':
            # Hard: Random field of many small obstacles (Swiss Cheese)
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            rng = np.random.RandomState(42)
            for _ in range(25):
                cx, cy = rng.randint(15, 85, 2)
                # Avoid placing obstacles on start/end
                if np.linalg.norm([cx - 5, cy - 50]) > 10 and np.linalg.norm([cx - 95, cy - 50]) > 10:
                    self.obstacles.append((cx, cy, 5))
                    
        elif name == 'Forest':
            # Very Hard: Dense random obstacles like navigating through trees
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            rng = np.random.RandomState(123)
            for _ in range(40):
                cx, cy = rng.randint(10, 90, 2)
                r = rng.uniform(2, 6)
                # Avoid start/end zones
                if np.linalg.norm([cx - 5, cy - 5]) > 15 and np.linalg.norm([cx - 95, cy - 95]) > 15:
                    self.obstacles.append((cx, cy, r))
                    
        elif name == 'Corridor':
            # Navigation through narrow corridor
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            # Top wall with gap
            for x in range(0, 45, 5):
                self.obstacles.append((x, 60, 4))
            for x in range(55, 100, 5):
                self.obstacles.append((x, 60, 4))
            # Bottom wall with gap
            for x in range(0, 45, 5):
                self.obstacles.append((x, 40, 4))
            for x in range(55, 100, 5):
                self.obstacles.append((x, 40, 4))

    def _segment_intersects_circle(self, p1, p2, circle):
        """Check if line segment p1-p2 hits a circular obstacle."""
        cx, cy, r = circle
        center = np.array([cx, cy])
        
        # Vector from p1 to p2
        d = p2 - p1
        f = p1 - center
        
        # Quadratic formula components for intersection
        a = np.dot(d, d)
        if a < 1e-10:  # Points are the same
            return np.linalg.norm(p1 - center) < r
            
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - r**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return False  # No intersection
            
        # Check if intersection points are actually on the segment
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a + 1e-6)
        t2 = (-b + discriminant) / (2*a + 1e-6)
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        
        # Also check if segment is completely inside circle
        if np.linalg.norm(p1 - center) < r or np.linalg.norm(p2 - center) < r:
            return True
            
        return False

    def decode(self, x):
        """Convert flat vector to list of points including start/end."""
        waypoints = x.reshape(-1, 2)
        # Clip to bounds
        waypoints = np.clip(waypoints, self.bounds[0], self.bounds[1])
        
        # Construct full path: Start -> Waypoints -> End
        path = np.vstack([self.start_pos, waypoints, self.end_pos])
        return path

    def count_collisions(self, path):
        """Count number of path segments that collide with obstacles."""
        collisions = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            for obs in self.obstacles:
                if self._segment_intersects_circle(p1, p2, (obs[0], obs[1], obs[2] + 1)):
                    collisions += 1
                    break  # Only count once per segment
        return collisions

    def evaluate(self, x):
        """
        Cost function = Total Length + Penalty (if hitting obstacle)
        Target: Minimize Cost
        """
        path = self.decode(x)
        total_dist = 0
        penalty = 0
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            # Add distance
            dist = np.linalg.norm(p2 - p1)
            total_dist += dist
            
            # Check collisions
            for obs in self.obstacles:
                # Add slight buffer to radius for safety
                if self._segment_intersects_circle(p1, p2, (obs[0], obs[1], obs[2] + 1)):
                    penalty += 1000  # Massive penalty for hitting wall
                    
        return total_dist + penalty

    def is_valid_path(self, x):
        """Check if path has no collisions."""
        path = self.decode(x)
        return self.count_collisions(path) == 0

    def visualize(self, algo_name, best_pos, best_score, save_path=None, show=False):
        """Draw the map and the best path found."""
        path = self.decode(best_pos)
        is_valid = self.is_valid_path(best_pos)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        
        status = "✓ Valid" if is_valid else "✗ Collision"
        ax.set_title(f"{algo_name} on {self.scenario_name}: Cost {best_score:.2f} ({status})")
        
        # Draw Obstacles
        for (cx, cy, r) in self.obstacles:
            circle = patches.Circle((cx, cy), r, color='firebrick', alpha=0.7)
            ax.add_patch(circle)
            
        # Draw Path
        path_color = 'blue' if is_valid else 'orange'
        ax.plot(path[:, 0], path[:, 1], '-o', color=path_color, linewidth=2, markersize=6, label='Path')
        
        # Draw waypoints
        for i, (wx, wy) in enumerate(path[1:-1]):
            ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(self.end_pos[0], self.end_pos[1], 'r*', markersize=15, label='End', zorder=5)
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


# ============================================
# BENCHMARK RESULTS CLASS
# ============================================

class PathfindingBenchmarkResults:
    """Store and analyze benchmark results"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.reference_costs = {}
        
    def add_result(self, algo_name, scenario, n_waypoints, cost, time_taken, is_valid):
        key = (scenario, n_waypoints)
        self.results[algo_name][key].append({
            'cost': cost,
            'time': time_taken,
            'valid': is_valid
        })
    
    def set_reference(self, scenario, n_waypoints, cost):
        self.reference_costs[(scenario, n_waypoints)] = cost
    
    def get_summary(self):
        """Generate summary statistics"""
        summary = {}
        for algo, configs in self.results.items():
            summary[algo] = {}
            for (scenario, n_wp), runs in configs.items():
                costs = [r['cost'] for r in runs]
                times = [r['time'] for r in runs]
                valid_count = sum(1 for r in runs if r['valid'])
                
                ref = self.reference_costs.get((scenario, n_wp), min(costs))
                gaps = [(c - ref) / ref * 100 if ref > 0 and c < 10000 else 100 for c in costs]
                
                summary[algo][(scenario, n_wp)] = {
                    'best': min(costs),
                    'mean': np.mean(costs),
                    'std': np.std(costs),
                    'worst': max(costs),
                    'mean_time': np.mean(times),
                    'mean_gap': np.mean(gaps),
                    'valid_ratio': valid_count / len(runs),
                    'runs': len(runs)
                }
        return summary
    
    def print_summary(self):
        """Print formatted summary table"""
        summary = self.get_summary()
        
        print("\n" + "=" * 110)
        print("                           PATHFINDING BENCHMARK RESULTS SUMMARY")
        print("=" * 110)
        
        # Group by problem configuration
        all_configs = set()
        for algo in summary:
            all_configs.update(summary[algo].keys())
        
        configs_sorted = sorted(all_configs, key=lambda x: (x[0], x[1]))
        
        for (scenario, n_wp) in configs_sorted:
            print(f"\n{'─' * 110}")
            print(f"  Scenario: {scenario.upper()} | Waypoints: {n_wp}")
            ref = self.reference_costs.get((scenario, n_wp), None)
            if ref:
                print(f"  Reference Cost: {ref:.2f}")
            print(f"{'─' * 110}")
            print(f"  {'Algorithm':<20} | {'Best':>10} | {'Mean':>10} | {'Std':>8} | {'Gap %':>8} | {'Valid %':>8} | {'Time (s)':>10}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
            
            # Sort algorithms by mean cost
            algo_results = []
            for algo in summary:
                if (scenario, n_wp) in summary[algo]:
                    algo_results.append((algo, summary[algo][(scenario, n_wp)]))
            
            algo_results.sort(key=lambda x: x[1]['mean'])
            
            for rank, (algo, stats) in enumerate(algo_results, 1):
                marker = "★" if rank == 1 else " "
                valid_pct = stats['valid_ratio'] * 100
                print(f"{marker} {algo:<20} | {stats['best']:>10.2f} | {stats['mean']:>10.2f} | "
                      f"{stats['std']:>8.2f} | {stats['mean_gap']:>7.2f}% | {valid_pct:>7.1f}% | {stats['mean_time']:>10.4f}")
        
        # Overall ranking
        print("\n" + "=" * 110)
        print("                              OVERALL ALGORITHM RANKING")
        print("=" * 110)
        
        # Calculate overall scores (lower is better) - penalize invalid paths
        algo_scores = defaultdict(list)
        algo_valid_rates = defaultdict(list)
        for algo in summary:
            for config, stats in summary[algo].items():
                # Use gap for valid paths, 100% penalty for invalid
                effective_gap = stats['mean_gap'] if stats['valid_ratio'] > 0.5 else 100 + stats['mean_gap']
                algo_scores[algo].append(effective_gap)
                algo_valid_rates[algo].append(stats['valid_ratio'])
        
        overall_ranking = [(algo, np.mean(gaps), np.mean(algo_valid_rates[algo])) 
                          for algo, gaps in algo_scores.items()]
        overall_ranking.sort(key=lambda x: x[1])
        
        print(f"\n  {'Rank':<6} | {'Algorithm':<25} | {'Avg Gap':>12} | {'Avg Valid %':>12}")
        print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*12}-+-{'-'*12}")
        for rank, (algo, avg_gap, avg_valid) in enumerate(overall_ranking, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank:<4} | {algo:<25} | {avg_gap:>11.2f}% | {avg_valid*100:>11.1f}%")
        
        print("\n" + "=" * 110)


class TeeStream:
    """Mirror writes to both terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_output_logging(log_file=None):
    """Redirect stdout/stderr to terminal + .txt file and return opened file handle."""
    PATHFINDING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PATHFINDING_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = PATHFINDING_LOGS_DIR / f'pathfinding_run_{timestamp}.txt'
    else:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PATHFINDING_LOGS_DIR / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, mode='w', encoding='utf-8')

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)

    print(f"[INFO] Logging benchmark output to: {log_path}")
    return log_handle, original_stdout, original_stderr


# ============================================
# BENCHMARK RUNNER
# ============================================

def run_benchmark():
    """
    Comprehensive Pathfinding Benchmark Suite
    
    Tests algorithms across:
    - Multiple scenarios (Sparse, Trap, Maze, Clutter, Forest)
    - Multiple waypoint counts
    - Multiple independent runs for statistical significance
    """
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "PATHFINDING BENCHMARK SUITE v2.0" + " " * 31 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 47 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Configuration
    SCENARIOS = ['Sparse', 'Trap', 'Maze', 'Clutter']
    WAYPOINT_COUNTS = [5, 8]
    RUNS_PER_CONFIG = 3
    MAX_ITER = 150
    POP_SIZE = 40
    
    results = PathfindingBenchmarkResults()
    
    # Define algorithms (create fresh instances for each run)
    def create_algos():
        algos = get_available_baseline_optimizers(pop_size=POP_SIZE)
        algos['Hagfish'] = HagfishPathfinder(POP_SIZE)
        return algos
    
    for scenario in SCENARIOS:
        for n_waypoints in WAYPOINT_COUNTS:
            print(f"\n{'━' * 90}")
            print(f"  Testing: Scenario = {scenario.upper()} | Waypoints = {n_waypoints}")
            print(f"{'━' * 90}")
            
            # Create problem instance
            problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
            
            # Compute reference (straight line distance - theoretical minimum)
            ref_cost = problem.min_distance
            results.set_reference(scenario, n_waypoints, ref_cost)
            print(f"  Theoretical Minimum (straight line): {ref_cost:.2f}")
            print()
            
            print(f"  {'Algorithm':<20} | {'Run 1':>10} | {'Run 2':>10} | {'Run 3':>10} | {'Best':>10} | {'Valid':>6} | {'Time':>8}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")
            
            # Test each algorithm
            algos = create_algos()
            for algo_name, algo in algos.items():
                run_costs = []
                run_times = []
                run_valid = []
                best_pos = None
                best_cost = float('inf')
                
                for run in range(RUNS_PER_CONFIG):
                    # Create fresh algorithm instance for each run
                    algo = create_algos()[algo_name]
                    
                    start = time.time()
                    
                    try:
                        pos, cost, _ = algo.optimize(
                            objective_fn=problem.evaluate,
                            bounds=problem.bounds,
                            dim=problem.dim,
                            max_iterations=MAX_ITER
                        )
                        
                        elapsed = time.time() - start
                        is_valid = problem.is_valid_path(pos)
                        
                        run_costs.append(cost)
                        run_times.append(elapsed)
                        run_valid.append(is_valid)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = pos
                        
                        results.add_result(algo_name, scenario, n_waypoints, cost, elapsed, is_valid)
                        
                    except Exception as e:
                        run_costs.append(float('inf'))
                        run_times.append(0)
                        run_valid.append(False)
                        print(f"  {algo_name:<20} | ERROR: {str(e)[:50]}")
                        continue
                
                # Print row
                if len(run_costs) == RUNS_PER_CONFIG:
                    c1, c2, c3 = run_costs
                    best = min(run_costs)
                    valid_count = sum(run_valid)
                    avg_time = np.mean(run_times)
                    valid_str = f"{valid_count}/{RUNS_PER_CONFIG}"
                    print(f"  {algo_name:<20} | {c1:>10.2f} | {c2:>10.2f} | {c3:>10.2f} | {best:>10.2f} | {valid_str:>6} | {avg_time:>7.3f}s")
                
                # Save visualization for best result
                if best_pos is not None:
                    save_path = PATHFINDING_PLOTS_DIR / f"path_{scenario}_{n_waypoints}wp_{algo_name}.png"
                    problem.visualize(algo_name, best_pos, best_cost, save_path=save_path)
    
    # Print final summary
    results.print_summary()
    
    return results


def run_quick_test():
    """Quick single-run test for debugging"""
    print("\n=== QUICK PATHFINDING TEST ===\n")
    
    scenario = 'Trap'
    n_waypoints = 5
    
    problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
    print(f"Testing: {scenario} scenario with {n_waypoints} waypoints")
    print(f"Start: {problem.start_pos}, End: {problem.end_pos}")
    print(f"Obstacles: {len(problem.obstacles)}")
    print(f"Theoretical min distance: {problem.min_distance:.2f}")
    print()
    
    # Test algorithms
    algos = get_available_baseline_optimizers(pop_size=40)
    algos['Hagfish'] = HagfishPathfinder(40)
    
    for name, algo in algos.items():
        start = time.time()
        pos, cost, _ = algo.optimize(
            objective_fn=problem.evaluate,
            bounds=problem.bounds,
            dim=problem.dim,
            max_iterations=100
        )
        elapsed = time.time() - start
        is_valid = problem.is_valid_path(pos)
        status = "✓" if is_valid else "✗"
        print(f"  {name:<20}: Cost = {cost:>8.2f}, Valid = {status}, Time = {elapsed:.3f}s")
        
        # Save visualization
        problem.visualize(name, pos, cost, save_path=PATHFINDING_PLOTS_DIR / f"quick_{scenario}_{name}.png")
    
    print("\nVisualization files saved!")


def run_scenario_demo():
    """Visualize all scenarios without running optimization"""
    print("\n=== SCENARIO VISUALIZATION ===\n")
    
    scenarios = ['Sparse', 'Trap', 'Maze', 'Clutter', 'Forest', 'Corridor']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        problem = RobotNavigation(n_waypoints=5, scenario=scenario)
        
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_title(f"{scenario} (Obstacles: {len(problem.obstacles)})")
        
        # Draw Obstacles
        for (cx, cy, r) in problem.obstacles:
            circle = patches.Circle((cx, cy), r, color='firebrick', alpha=0.7)
            ax.add_patch(circle)
        
        # Draw start/end
        ax.plot(problem.start_pos[0], problem.start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(problem.end_pos[0], problem.end_pos[1], 'r*', markersize=12, label='End')
        
        # Draw straight line (theoretical path)
        ax.plot([problem.start_pos[0], problem.end_pos[0]], 
                [problem.start_pos[1], problem.end_pos[1]], 
                '--', color='gray', alpha=0.5, label='Direct')
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    overview_path = PATHFINDING_PLOTS_DIR / 'scenarios_overview.png'
    plt.savefig(overview_path, dpi=150)
    print(f"Saved: {overview_path}")
    plt.close()


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathfinding Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    parser.add_argument('--demo', action='store_true', help='Visualize scenarios only')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Optional log filename/path for .txt output (default: timestamped file in output/logs/pathfinding)')
    
    args = parser.parse_args()
    
    log_handle, original_stdout, original_stderr = setup_output_logging(args.log_file)
    try:
        if args.quick:
            run_quick_test()
        elif args.demo:
            run_scenario_demo()
        else:
            run_benchmark()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.flush()
        log_handle.close()
