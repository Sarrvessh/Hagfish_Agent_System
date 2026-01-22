import numpy as np

class PSO:
    """Kennedy & Eberhart (1995) - Particle Swarm Optimization"""
    def __init__(self, pop_size=30, c1=2.0, c2=2.0, w=0.7):
        self.pop_size = pop_size
        self.c1, self.c2, self.w = c1, c2, w

    def optimize(self, objective_fn, bounds, dim, max_iterations=200):
        # Handle both scalar and array bounds
        lb = np.array(bounds[0]) if hasattr(bounds[0], '__len__') else np.full(dim, bounds[0])
        ub = np.array(bounds[1]) if hasattr(bounds[1], '__len__') else np.full(dim, bounds[1])
        
        positions = np.random.uniform(lb, ub, (self.pop_size, dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        pbest_positions = positions.copy()
        pbest_fitness = np.array([objective_fn(p) for p in positions])
        gbest_idx = np.argmin(pbest_fitness)
        convergence = []

        for iteration in range(max_iterations):
            w = self.w * (1 - iteration / max_iterations)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    w * velocities[i] +
                    self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                    self.c2 * r2 * (positions[gbest_idx] - positions[i])
                )

                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
                fitness_i = objective_fn(positions[i])

                if fitness_i < pbest_fitness[i]:
                    pbest_fitness[i] = fitness_i
                    pbest_positions[i] = positions[i]

                    if fitness_i < pbest_fitness[gbest_idx]:
                        gbest_idx = i

            convergence.append(pbest_fitness[gbest_idx])

        return positions[gbest_idx], pbest_fitness[gbest_idx], convergence

class GWO:
    """Mirjalili et al. (2014) - Grey Wolf Optimizer"""
    def __init__(self, pop_size=30):
        self.pop_size = pop_size

    def optimize(self, objective_fn, bounds, dim, max_iterations=200):
        # Handle both scalar and array bounds
        lb = np.array(bounds[0]) if hasattr(bounds[0], '__len__') else np.full(dim, bounds[0])
        ub = np.array(bounds[1]) if hasattr(bounds[1], '__len__') else np.full(dim, bounds[1])
        
        positions = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.array([objective_fn(p) for p in positions])

        sorted_indices = np.argsort(fitness)
        alpha_pos = positions[sorted_indices[0]].copy()
        beta_pos = positions[sorted_indices[1]].copy() if self.pop_size > 1 else alpha_pos.copy()
        delta_pos = positions[sorted_indices[2]].copy() if self.pop_size > 2 else alpha_pos.copy()
        convergence = []

        for iteration in range(max_iterations):
            a = 2 - 2 * (iteration / max_iterations)

            for i in range(self.pop_size):
                for d in range(dim):
                    r1, r2 = np.random.rand(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    X1 = alpha_pos[d] - A1 * abs(C1 * alpha_pos[d] - positions[i, d])

                    r1, r2 = np.random.rand(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    X2 = beta_pos[d] - A2 * abs(C2 * beta_pos[d] - positions[i, d])

                    r1, r2 = np.random.rand(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    X3 = delta_pos[d] - A3 * abs(C3 * delta_pos[d] - positions[i, d])

                    positions[i, d] = np.clip((X1 + X2 + X3) / 3, lb[d], ub[d])

            fitness = np.array([objective_fn(p) for p in positions])
            sorted_indices = np.argsort(fitness)

            alpha_pos = positions[sorted_indices[0]].copy()
            if self.pop_size > 1:
                beta_pos = positions[sorted_indices[1]].copy()
            if self.pop_size > 2:
                delta_pos = positions[sorted_indices[2]].copy()

            convergence.append(fitness[sorted_indices[0]])

        return alpha_pos, fitness[sorted_indices[0]], convergence

class GA:
    """Holland (1975) - Genetic Algorithm"""
    def __init__(self, pop_size=30, mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(self, objective_fn, bounds, dim, max_iterations=200):
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, dim))
        convergence = []

        for iteration in range(max_iterations):
            fitness = np.array([objective_fn(ind) for ind in population])
            best_idx = np.argmin(fitness)
            convergence.append(fitness[best_idx])

            # Selection via fitness-based probability
            probabilities = 1.0 / (fitness + 1e-10)
            probabilities /= probabilities.sum()

            new_population = []
            for _ in range(self.pop_size):
                parent1_idx, parent2_idx = np.random.choice(
                    self.pop_size, 2, p=probabilities, replace=False
                )

                # Crossover
                child = (population[parent1_idx] + population[parent2_idx]) / 2

                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, dim)
                    child = child + mutation

                child = np.clip(child, bounds[0], bounds[1])
                new_population.append(child)

            population = np.array(new_population)

        fitness = np.array([objective_fn(ind) for ind in population])
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], convergence

class DE:
    """Storn & Price (1997) - Differential Evolution"""
    def __init__(self, pop_size=30, F=0.8, CR=0.9):
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability

    def optimize(self, objective_fn, bounds, dim, max_iterations=200):
        # Handle both scalar and array bounds
        lb = np.array(bounds[0]) if hasattr(bounds[0], '__len__') else np.full(dim, bounds[0])
        ub = np.array(bounds[1]) if hasattr(bounds[1], '__len__') else np.full(dim, bounds[1])
        
        population = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.array([objective_fn(p) for p in population])
        convergence = []

        for iteration in range(max_iterations):
            for i in range(self.pop_size):
                # Select three random distinct individuals
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)

                # Mutation: create mutant vector
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover: create trial vector
                trial = population[i].copy()
                for d in range(dim):
                    if np.random.rand() < self.CR:
                        trial[d] = mutant[d]

                # Selection: accept if better
                trial_fitness = objective_fn(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            convergence.append(np.min(fitness))

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], convergence

class FA:
    """Yang (2008) - Firefly Algorithm"""
    def __init__(self, pop_size=30, alpha=0.2, beta=1.0, gamma=1.0):
        self.pop_size = pop_size
        self.alpha = alpha  # Randomization parameter
        self.beta = beta  # Light absorption coefficient
        self.gamma = gamma  # Exponential decay rate

    def optimize(self, objective_fn, bounds, dim, max_iterations=200):
        positions = np.random.uniform(bounds[0], bounds[1], (self.pop_size, dim))
        fitness = np.array([objective_fn(p) for p in positions])
        convergence = []

        for iteration in range(max_iterations):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(positions[i] - positions[j])
                        beta = self.beta * np.exp(-self.gamma * distance ** 2)

                        positions[i] += (
                            beta * (positions[j] - positions[i]) +
                            self.alpha * (np.random.rand(dim) - 0.5)
                        )
                        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

            fitness = np.array([objective_fn(p) for p in positions])
            convergence.append(np.min(fitness))

        best_idx = np.argmin(fitness)
        return positions[best_idx], fitness[best_idx], convergence