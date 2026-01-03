import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ----- Legacy optimizer (kept for compatibility) -----
# NOTE: The agentic ML system uses a supervised dataset and trains a model inside SolverAgent.
# The legacy optimizer implementation below is preserved for completeness and is not used for ML data loading.
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class OptimizedHagfishSlimeTSP:
    """
    Legacy combinatorial optimizer (originally designed for route optimization).

    This implementation is preserved for historical completeness only. The
    active ML experiment framework loads supervised data and trains an ML
    model inside `SolverAgent` (see below). The implementation contains
    several heuristic optimizations (initialization heuristics, local search,
    adaptive operators) but should be considered a legacy component for this
    repository's ML-focused experiments.
    """

    def __init__(self, dist_matrix: np.ndarray, pop_size: int = 120, 
                 max_iter: int = 800, elite_size: int = 8):
        self.D = dist_matrix
        self.n = dist_matrix.shape[0]
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elite_size = elite_size
        
        # Population and fitness
        self.X = self._init_pop_smart()
        self.F = self._eval(self.X)
        
        # Slime matrix
        self.S = np.zeros((self.n, self.n))
        self.S_decay_rate = 0.80
        
        # Global best
        best_idx = np.argmin(self.F)
        self.best_x = self.X[best_idx].copy()
        self.best_f = self.F[best_idx]
        self.best_history = [self.best_f]
        
        # Stagnation
        self.stag_count = 0
        self.restart_threshold = 100

    # ========== SMART INITIALIZATION ==========

    def _init_pop_smart(self) -> np.ndarray:
        """Smarter population initialization using savings algorithm + random."""
        X = []
        
        # 1. Nearest neighbor from multiple starts
        nn_count = min(8, self.pop_size // 8)
        for start in range(nn_count):
            X.append(self._nearest_neighbor(start * max(1, self.n // nn_count)))
        
        # 2. Greedy savings algorithm (Christofides-inspired)
        X.append(self._savings_tour())
        
        # 3. Farthest insertion heuristic
        X.append(self._farthest_insertion())
        
        # 4. Random + perturbed versions
        while len(X) < self.pop_size:
            if len(X) < self.pop_size // 2:
                # Perturb existing good solutions
                X.append(self._perturb(X[len(X) % len(X[:5])], intensity=0.3))
            else:
                # Full random
                X.append(np.random.permutation(self.n))
        
        return np.array(X[:self.pop_size])

    def _nearest_neighbor(self, start: int) -> np.ndarray:
        """Nearest neighbor from start city."""
        unvisited = set(range(self.n))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            nearest = min(unvisited, key=lambda x: self.D[last, x])
            tour.append(nearest)
            unvisited.remove(nearest)
        return np.array(tour)

    def _savings_tour(self) -> np.ndarray:
        """Savings algorithm (fast greedy TSP heuristic)."""
        start_city = 0
        tour = [start_city]
        unvisited = set(range(1, self.n))
        
        while unvisited:
            last = tour[-1]
            # Compute savings: d(last, next) + d(next, tour[0]) - d(last, tour[0])
            savings = {}
            for city in unvisited:
                s = self.D[last, city] + self.D[city, start_city] - self.D[last, start_city]
                savings[city] = -s  # negative for max-heap behavior
            best_city = min(unvisited, key=lambda x: savings[x])
            tour.append(best_city)
            unvisited.remove(best_city)
        
        return np.array(tour)

    def _farthest_insertion(self) -> np.ndarray:
        """Farthest insertion heuristic."""
        start = 0
        tour = [start]
        unvisited = set(range(1, self.n))
        
        # Add farthest from start
        farthest = max(unvisited, key=lambda x: self.D[start, x])
        tour.append(farthest)
        unvisited.remove(farthest)
        
        while unvisited:
            # Find farthest point from tour
            farthest = max(unvisited, 
                         key=lambda x: min(self.D[x, city] for city in tour))
            
            # Find best insertion position
            best_pos = 1
            best_cost = float('inf')
            for pos in range(len(tour)):
                i, j = tour[pos], tour[(pos + 1) % len(tour)]
                cost = self.D[i, farthest] + self.D[farthest, j] - self.D[i, j]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos + 1
            
            tour.insert(best_pos, farthest)
            unvisited.remove(farthest)
        
        return np.array(tour)

    # ========== EVALUATION ==========

    def _eval(self, X: np.ndarray) -> np.ndarray:
        """Vectorized evaluation."""
        return np.sum(self.D[X[:, :-1], X[:, 1:]], axis=1) + self.D[X[:, -1], X[:, 0]]

    def _eval_single(self, tour: np.ndarray) -> float:
        """Single tour distance."""
        return np.sum(self.D[tour[:-1], tour[1:]]) + self.D[tour[-1], tour[0]]

    # ========== SLIME MECHANICS ==========

    def _mark_slime(self, tour: np.ndarray, strength: float = 0.3):
        """Mark edges with slime."""
        edges_from = tour[:-1]
        edges_to = tour[1:]
        self.S[edges_from, edges_to] += strength
        self.S[edges_to, edges_from] += strength
        self.S[tour[-1], tour[0]] += strength
        self.S[tour[0], tour[-1]] += strength

    def _slime_penalty(self, tour: np.ndarray) -> float:
        """Slime penalty for tour."""
        edges_from = tour[:-1]
        edges_to = tour[1:]
        return np.sum(self.S[edges_from, edges_to]) + self.S[tour[-1], tour[0]] + self.S[tour[0], tour[-1]]

    # ========== LOCAL SEARCH (3-OPT) ==========

    def _three_opt(self, tour: np.ndarray, prob_improve: float = 0.6) -> np.ndarray:
        """Efficient 3-opt with early termination."""
        new_tour = tour.copy()
        improved = True
        attempts = 0
        max_attempts = min(self.n, 20)
        
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            
            for _ in range(3):  # Try a few random 3-opt moves
                i, j, k = sorted(np.random.choice(self.n, 3, replace=False))
                if j - i < 2 or k - j < 2:
                    continue
                
                # Current cost of three segments
                seg1_cost = self.D[tour[i-1], tour[i]] + self.D[tour[j], tour[j+1]]
                seg2_cost = self.D[tour[j-1], tour[j]] + self.D[tour[k], tour[k+1 if k+1 < self.n else 0]]
                
                # Try reversal
                candidate = tour.copy()
                candidate[i:j+1] = candidate[i:j+1][::-1]
                
                cand_cost = self._eval_single(candidate)
                if cand_cost < self._eval_single(new_tour):
                    new_tour = candidate
                    improved = True
                    break
        
        return new_tour

    def _two_opt_fast(self, tour: np.ndarray) -> np.ndarray:
        """Fast 2-opt (simpler but quicker than 3-opt)."""
        i, j = sorted(np.random.choice(self.n, 2, replace=False))
        new_tour = tour.copy()
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour

    def _crossover_ox(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order crossover."""
        size = self.n
        child = np.full(size, -1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child[start:end] = p1[start:end]
        in_child = np.zeros(size, dtype=bool)
        in_child[child[start:end]] = True
        pos = end
        for city in np.roll(p2, -end):
            if not in_child[city]:
                if pos >= size:
                    pos = 0
                child[pos] = city
                pos += 1
        return child

    # ========== OPTIMIZATION LOOP ==========

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Main optimization loop with adaptive operators.

        Note: This is part of the legacy optimizer implementation and is not
        involved in the ML training pipeline. It is preserved for completeness
        and historical reference only.
        """
        for it in range(self.max_iter):
            new_X = []
            new_F = []

            # Elite preservation
            elite_idx = np.argsort(self.F)[:self.elite_size]
            for idx in elite_idx:
                new_X.append(self.X[idx].copy())
                new_F.append(self.F[idx])

            # Evolve population
            for i in range(self.pop_size - self.elite_size):
                # Adaptive local search: more 3-opt early, more 2-opt late
                progress = it / self.max_iter
                if np.random.random() < (0.7 - 0.4 * progress):  # 70% -> 30% 3-opt
                    cand = self._three_opt(self.X[i])
                else:
                    cand = self._two_opt_fast(self.X[i])
                
                f_cand = self._eval_single(cand)

                # Adaptive crossover: higher prob early
                if np.random.random() < (0.4 - 0.2 * progress):
                    cand2 = self._crossover_ox(self.X[i], self.best_x)
                    f2 = self._eval_single(cand2)
                    if f2 < f_cand:
                        cand, f_cand = cand2, f2

                # Accept if better
                if f_cand < self.F[i]:
                    new_X.append(cand)
                    new_F.append(f_cand)
                else:
                    new_X.append(self.X[i].copy())
                    new_F.append(self.F[i])

            self.X = np.array(new_X)
            self.F = np.array(new_F)

            # Update global best
            curr_best_idx = np.argmin(self.F)
            if self.F[curr_best_idx] < self.best_f:
                self.best_f = self.F[curr_best_idx]
                self.best_x = self.X[curr_best_idx].copy()
                self.stag_count = 0
            else:
                self.stag_count += 1

            self.best_history.append(self.best_f)

            # Slime management
            if it % 15 == 0 and it > 50:  # Start after warmup
                worst_k = max(1, int(self.pop_size * 0.15))
                worst_idx = np.argsort(self.F)[-worst_k:]
                for idx in worst_idx:
                    self._mark_slime(self.X[idx], strength=0.25)

            # Slime decay with schedule
            if it % 30 == 0:
                self.S *= self.S_decay_rate

            # Adaptive restart: more aggressive later
            restart_thresh_adaptive = self.restart_threshold * (1 - 0.3 * progress)
            if self.stag_count >= int(restart_thresh_adaptive):
                num_restart = max(2, int(self.pop_size * 0.25))
                restart_idx = np.argsort(self.F)[-num_restart:]
                for idx in restart_idx:
                    self.X[idx] = self._perturb(self.best_x, intensity=0.5)
                self.F = self._eval(self.X)
                self.stag_count = 0

            if (it + 1) % 100 == 0:
                print(f"Iter {it+1:4d} | Best: {self.best_f:.2f} | Slime: {np.sum(self.S > 0.01):.0f}")

        return self.best_x, self.best_f, self.best_history

    def _perturb(self, tour: np.ndarray, intensity: float) -> np.ndarray:
        """Smart perturbation."""
        new_tour = tour.copy()
        n_swaps = max(1, int(self.n * intensity))
        for _ in range(n_swaps):
            i, j = np.random.choice(self.n, 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour


# ============================================
# BENCHMARK
# ============================================

def benchmark_tsp(n_cities: int = 100, seed: int = 42):
    """Generate and solve TSP instance."""
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    
    D = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d

    print(f"\n{'='*60}")
    print(f"Optimized Hagfish Slime TSP - {n_cities} cities")
    print(f"{'='*60}")

    # Adaptive parameters based on problem size
    pop_size = min(150, max(80, n_cities // 2))
    max_iter = min(1000, max(500, n_cities * 5))
    
    solver = OptimizedHagfishSlimeTSP(D, pop_size=pop_size, max_iter=max_iter)
    
    start = time.time()
    best_tour, best_dist, history = solver.optimize()
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Solution: {best_dist:.2f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Iterations: {max_iter}")
    print(f"Population: {pop_size}")
    print(f"{'='*60}\n")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence
    ax1.plot(history, linewidth=2, color='steelblue')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Distance (log)', fontsize=12)
    ax1.set_title('Convergence Curve', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Tour
    tour_coords = coords[best_tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])
    ax2.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-', linewidth=1, alpha=0.7)
    ax2.scatter(tour_coords[:-1, 0], tour_coords[:-1, 1], c='blue', s=30, zorder=5)
    ax2.scatter([tour_coords[0, 0]], [tour_coords[0, 1]], c='green', s=100, marker='*', zorder=6, label='Start')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title(f'Best Tour (Distance: {best_dist:.2f})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return best_dist, elapsed

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Legacy optimizer benchmark (small)")
    print("="*60)
    benchmark_tsp(n_cities=30, seed=42)

    print("\n" + "="*60)
    print("Legacy optimizer benchmark (medium)")
    print("="*60)
    benchmark_tsp(n_cities=100, seed=123)

    print("\n" + "="*60)
    print("Legacy optimizer benchmark (large)")
    print("="*60)
    benchmark_tsp(n_cities=150, seed=123)

# ----- Solver Agent wrapper -----

class SolverAgent:
    """A deterministic wrapper around OptimizedHagfishSlimeTSP.

    This class treats OptimizedHagfishSlimeTSP as a black-box solver agent and
    provides a consistent interface to run optimization with explicit control
    of randomness via `random_seed`.
    """

    def __init__(self, dist_matrix: np.ndarray):
        """SolverAgent wrapper initialization (ML training environment).

        This constructor preserves the original `dist_matrix` signature for
        backward compatibility, but the active ML experiment loads a real
        supervised dataset (the `breast_cancer` dataset from scikit-learn)
        and prepares deterministic training and validation splits.

        Semantics (resource -> ML mapping):
          - `pop_size`  -> batch size (number of samples per mini-batch)
          - `max_iter`  -> number of training epochs
          - `elite_size`-> interpreted as a regularization/reserved capacity

        The training dataset and preprocessed splits are stored as instance
        attributes (`X_train`, `X_val`, `y_train`, `y_val`) so that multiple
        runs remain deterministic and reproducible.
        """
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]

        # Load real ML dataset and prepare train/validation splits
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_val = scaler.transform(X_val)
        self.y_train = y_train
        self.y_val = y_val
        # Keep scaler if needed externally
        self.scaler = scaler

    def run(self, pop_size: int = 120, max_iter: int = 800, elite_size: int = 8,
            random_seed: int = 0) -> Dict[str, Any]:
        """Run a supervised training workload using the allocated training budget.

        Resource mapping (for clarity):
          - `pop_size` : batch size (samples per mini-batch during training)
          - `max_iter` : number of training epochs
          - `elite_size`: regularization / reserved capacity (kept for compatibility)

        The routine trains a supervised model on a held-out validation set and
        returns per-epoch validation metrics. The returned `best_metric` is the
        best validation accuracy observed across epochs.
        """
        # Deterministic control
        rs = np.random.RandomState(int(random_seed))

        # Map resources
        batch_size = int(pop_size)
        n_epochs = int(max_iter)
        C = max(1.0 / max(1, int(elite_size)), 1e-6)  # regularization inverse mapping

        history: List[float] = []
        best_metric = float("-inf")

        # Use SGDClassifier with `partial_fit` for clean epoch-wise training
        # This avoids solver convergence warnings while allowing mini-batch updates
        from sklearn.linear_model import SGDClassifier

        clf = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1, tol=None,
                            random_state=int(random_seed))

        start = time.time()

        classes = np.unique(self.y_train)
        n_samples = self.X_train.shape[0]
        batch_size = max(1, int(batch_size))

        # Train for n_epochs using partial_fit and mini-batches
        for epoch in range(n_epochs):
            # Deterministic batching order
            indices = np.arange(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = self.X_train[indices[start_idx:end_idx]]
                y_batch = self.y_train[indices[start_idx:end_idx]]
                if epoch == 0 and start_idx == 0:
                    clf.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    clf.partial_fit(X_batch, y_batch)

            score = float(clf.score(self.X_val, self.y_val))
            history.append(score)
            if score > best_metric:
                best_metric = score

        elapsed = time.time() - start

        # Provide a lightweight, backward-compatible model metadata field
        # under the `best_tour` key so downstream code expecting this field
        # (from the legacy TSP framing) can still display model information
        # (e.g., number of learned features) without changing existing APIs.
        model_info = list(range(self.X_train.shape[1]))

        return {
            "best_metric": float(best_metric),
            "training_time": float(elapsed),
            "history": list(map(float, history)),
            "params": {
                "pop_size": int(pop_size),
                "max_iter": int(max_iter),
                "elite_size": int(elite_size),
                "random_seed": int(random_seed),
            },
            "best_tour": model_info,
        }
