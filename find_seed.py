"""
Hagfish Slime Defense Optimizer (High-Performance Version)

Optimizations:
  1. Hybrid Data Structures: Uses NumPy for vector ops, Python Lists for tight loops.
  2. Fast 2-opt: Optimized logic to remove crossing paths instantly.
  3. Aggressive Slime: Stronger repulsion to force exploration.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set

class HagfishTSPSolver:
    def __init__(self, dist_matrix: np.ndarray, pop_size: int = 20, 
                 decay_rate: float = 0.1, verbose: bool = True):
        self.D_np = dist_matrix
        # CRITICAL OPTIMIZATION: Convert to Python list for fast scalar access in loops
        self.D = dist_matrix.tolist() 
        self.n = dist_matrix.shape[0]
        self.pop_size = pop_size
        self.decay_rate = decay_rate
        self.verbose = verbose
        
        # State
        self.population = []
        self.fitness = []
        self.slime_edges = set()
        self.elite_tour = None
        self.elite_fitness = float('inf')
        self.slime_intensity = 0.5
        self.stagnation_count = 0
        self.burst_count = 0
        self.history = []

    def _eval(self, tour: List[int]) -> float:
        """Fast evaluation using list indexing."""
        d = 0.0
        n = self.n
        matrix = self.D
        for i in range(n - 1):
            d += matrix[tour[i]][tour[i+1]]
        d += matrix[tour[-1]][tour[0]]
        return d

    def _fast_2opt(self, tour: List[int]) -> List[int]:
        """
        High-Performance 2-opt Local Search.
        Uses Python lists to avoid NumPy scalar overhead.
        """
        n = self.n
        matrix = self.D
        improved = True
        best_tour = tour[:]
        
        # Limit passes to prevent infinite loops, but usually converges fast
        for _ in range(5): 
            if not improved: break
            improved = False
            
            for i in range(n - 2):
                # Cache start node
                u = best_tour[i]
                u_next = best_tour[i+1]
                dist_u_un = matrix[u][u_next]
                
                for j in range(i + 2, n):
                    # Wrap-around handling
                    if j == n - 1 and i == 0: continue
                    
                    v = best_tour[j]
                    v_next = best_tour[(j + 1) % n]
                    
                    # Cost of current edges
                    curr_dist = dist_u_un + matrix[v][v_next]
                    # Cost of new edges
                    new_dist = matrix[u][v] + matrix[u_next][v_next]
                    
                    if new_dist < curr_dist - 1e-9:
                        # Swap: Reverse the segment between i+1 and j
                        best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                        improved = True
                        # Restart inner loop (First Improvement Strategy is faster)
                        break 
        return best_tour

    def _initialize(self):
        self.population = []
        self.fitness = []
        # Generate random start
        for _ in range(self.pop_size):
            # NumPy permutation is fine for generation
            tour = np.random.permutation(self.n).tolist()
            # Apply 2-opt immediately to start strong
            tour = self._fast_2opt(tour)
            f = self._eval(tour)
            
            self.population.append(tour)
            self.fitness.append(f)
            
            if f < self.elite_fitness:
                self.elite_fitness = f
                self.elite_tour = tour[:]

    def _get_slime_penalty(self, tour: List[int]) -> float:
        """Calculate penalty factor (Logic: Lower factor = more slime)."""
        if not self.slime_edges: return 1.0
        
        # Count overlaps
        hits = 0
        n = self.n
        for i in range(n):
            u, v = tour[i], tour[(i+1)%n]
            if u > v: u, v = v, u
            if (u, v) in self.slime_edges:
                hits += 1
        
        ratio = hits / n
        return 1.0 - (self.slime_intensity * ratio)

    def _select_parent(self) -> List[int]:
        """Tournament selection with Slime Logic."""
        # Sample 3 random candidates
        indices = np.random.randint(0, self.pop_size, 3)
        
        best_score = float('inf')
        best_idx = -1
        
        for i in indices:
            f = self.fitness[i]
            p = self._get_slime_penalty(self.population[i])
            # Logic: Score = Fitness / Penalty. 
            # If slimed (p < 1.0), score goes UP (bad).
            score = f / max(0.01, p)
            
            if score < best_score:
                best_score = score
                best_idx = i
                
        return self.population[best_idx][:]

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """Order Crossover (OX) using Lists."""
        n = self.n
        start = np.random.randint(0, n)
        end = np.random.randint(start + 1, n + 1)
        
        child = [-1] * n
        child[start:end] = p1[start:end]
        
        # Set for O(1) lookup
        in_child = set(child[start:end])
        
        curr = end % n
        p2_idx = end % n
        
        count = 0
        while count < n:
            city = p2[p2_idx]
            if city not in in_child:
                if curr >= n: curr = 0
                if child[curr] == -1:
                    child[curr] = city
                    curr += 1
            p2_idx = (p2_idx + 1) % n
            count += 1
            
        return child

    def _deposit_slime(self, tour: List[int]):
        """Mark edges of the worst tour."""
        n = self.n
        for i in range(n):
            u, v = tour[i], tour[(i+1)%n]
            if u > v: u, v = v, u
            self.slime_edges.add((u, v))

    def _burst(self):
        """Explosion mechanism."""
        self.slime_intensity = min(0.9, self.slime_intensity + 0.2)
        # Create a mutated version of elite
        burst = self.elite_tour[:]
        # Heavy mutation
        for _ in range(self.n // 4):
            i, j = np.random.randint(0, self.n, 2)
            burst[i], burst[j] = burst[j], burst[i]
        return self._fast_2opt(burst)

    def solve(self, max_iter: int = 100):
        start_time = time.time()
        self._initialize()
        
        interval_bursts = 0
        
        for it in range(max_iter):
            # Decay
            self.slime_intensity *= (1.0 - self.decay_rate)
            
            # Stagnation
            current_best = min(self.fitness)
            if current_best >= self.elite_fitness - 1e-6:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                self.elite_fitness = current_best
            
            # BURST
            if self.stagnation_count >= 10:
                burst_tour = self._burst()
                f = self._eval(burst_tour)
                # Kill worst
                worst = np.argmax(self.fitness)
                self.population[worst] = burst_tour
                self.fitness[worst] = f
                
                self.stagnation_count = 0
                self.burst_count += 1
                interval_bursts += 1
            
            # SLIME
            worst = np.argmax(self.fitness)
            if self.fitness[worst] > self.elite_fitness * 1.1:
                self._deposit_slime(self.population[worst])
            
            # EVOLVE
            new_pop = []
            new_fit = []
            
            # Elitism (Keep top 2)
            indices = np.argsort(self.fitness)
            for i in range(2):
                idx = indices[i]
                new_pop.append(self.population[idx][:])
                new_fit.append(self.fitness[idx])
                
            while len(new_pop) < self.pop_size:
                p1 = self._select_parent()
                p2 = self._select_parent()
                
                child = self._crossover(p1, p2)
                
                # Mutation (Swap)
                if np.random.random() < 0.1:
                    i, j = np.random.randint(0, self.n, 2)
                    child[i], child[j] = child[j], child[i]
                
                # Local Search
                child = self._fast_2opt(child)
                
                new_pop.append(child)
                new_fit.append(self._eval(child))
                
            self.population = new_pop
            self.fitness = new_fit
            
            # Check Global Elite
            gen_best_idx = np.argmin(self.fitness)
            if self.fitness[gen_best_idx] < self.elite_fitness:
                self.elite_fitness = self.fitness[gen_best_idx]
                self.elite_tour = self.population[gen_best_idx][:]
            
            self.history.append(self.elite_fitness)
            
            if self.verbose and (it+1) % 50 == 0:
                print(f"Iter {it+1:4d} | Best: {self.elite_fitness:.2f} | Bursts: {interval_bursts}")
                interval_bursts = 0

        total_time = time.time() - start_time
        return {
            "dist": self.elite_fitness,
            "time": total_time,
            "tour": self.elite_tour
        }

# ==========================================
# BENCHMARK
# ==========================================
def create_scaled_tsp(n, scale=100.0, seed=42):
    np.random.seed(seed)
    cities = np.random.rand(n, 2) * scale
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(cities[i]-cities[j])
    return D

if __name__ == "__main__":
    print("="*60)
    print("HIGH-PERFORMANCE HAGFISH OPTIMIZER")
    print("="*60)
    
    # Matching your previous output scales for comparison
    
    # 30 Cities
    D1 = create_scaled_tsp(30, scale=35.0, seed=175)
    s1 = HagfishTSPSolver(D1, pop_size=20)
    r1 = s1.solve(300)
    print(f"[30 Cities] Time: {r1['time']:.4f}s | Dist: {r1['dist']:.2f}")

    # 50 Cities
    D2 = create_scaled_tsp(50, scale=38.0, seed=250)
    s2 = HagfishTSPSolver(D2, pop_size=30)
    r2 = s2.solve(300)
    print(f"[50 Cities] Time: {r2['time']:.4f}s | Dist: {r2['dist']:.2f}")

    # 100 Cities
    D3 = create_scaled_tsp(100, scale=42.0, seed=420)
    s3 = HagfishTSPSolver(D3, pop_size=40)
    r3 = s3.solve(300)
    print(f"[100 Cities] Time: {r3['time']:.4f}s | Dist: {r3['dist']:.2f}")