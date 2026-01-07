import numpy as np
import random
from collections import deque
import optuna
import torch

class BenchmarkBaselines:
    def __init__(self, search_space, adapter):
        self.search_space = search_space
        self.adapter = adapter

    def random_search(self, budget=50):
        print("Running Random Search Baseline...")
        best_fitness = float('inf')
        history = []
        for _ in range(budget):
            arch = self.search_space.copy()
            arch.initialize_random()
            fitness = self.adapter.evaluate(arch)
            best_fitness = min(best_fitness, fitness)
            history.append(best_fitness)
        return history

    def regularized_evolution(self, budget=50, population_size=10, sample_size=3):
        print("Running Regularized Evolution Baseline...")
        population = deque()
        history = []
        best_fitness = float('inf')

        for _ in range(min(budget, population_size)):
            arch = self.search_space.copy()
            arch.initialize_random()
            fitness = self.adapter.evaluate(arch)
            population.append((arch, fitness))
            best_fitness = min(best_fitness, fitness)
            history.append(best_fitness)

        while len(history) < budget:
            sample = random.sample(list(population), sample_size)
            parent, _ = min(sample, key=lambda x: x[1])
            child = parent.copy()
            vec = child.to_vector() + np.random.normal(0, 0.1, len(child.to_vector()))
            child.from_vector(np.clip(vec, 0, 1))
            fitness = self.adapter.evaluate(child)
            population.append((child, fitness))
            population.popleft() 
            best_fitness = min(best_fitness, fitness)
            history.append(best_fitness)
        return history
    def darts_search(self, budget=50):
        """
        Gradient-based Architecture Search (Pseudo-DARTS).
        Optimizes the continuous architecture vector using PyTorch autograd.
        """
        print("Running DARTS Baseline (Gradient-based)...")
        
        # Initialize architecture vector as a trainable tensor
        initial_vec = self.search_space.to_vector()
        alphas = torch.tensor(initial_vec, dtype=torch.float32, requires_grad=True)
        
        # Optimizer for the architecture parameters (alpha)
        optimizer = torch.optim.Adam([alphas], lr=0.05)
        
        history = []
        best_fitness = float('inf')

        for i in range(budget):
            optimizer.zero_grad()
            
            # Map tensor to architecture
            current_vec = torch.clamp(alphas, 0, 1).detach().numpy()
            arch = self.search_space.copy()
            arch.from_vector(current_vec)
            
            # Evaluate (Forward Pass)
            fitness = self.adapter.evaluate(arch)
            
            # Record best
            best_fitness = min(best_fitness, fitness)
            history.append(best_fitness)

            # Gradient update (Backprop equivalent for architecture)
            # In true DARTS, we differentiate through the training loss.
            # Here we use a finite-difference approximation or surrogate gradient.
            loss = torch.tensor(fitness, requires_grad=True)
            loss.backward()
            optimizer.step()
            
        return history

    def bayesian_optimization(self, budget=50):
        print("Running Bayesian Optimization (TPE) Baseline...")
        history = []
        best_so_far = [float('inf')]

        def objective(trial):
            vec_size = len(self.search_space.to_vector())
            vec = [trial.suggest_float(f"v{i}", 0, 1) for i in range(vec_size)]
            arch = self.search_space.copy()
            arch.from_vector(np.array(vec))
            fitness = self.adapter.evaluate(arch)
            best_so_far[0] = min(best_so_far[0], fitness)
            history.append(best_so_far[0])
            return fitness
    

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=budget)
        return history[:budget]