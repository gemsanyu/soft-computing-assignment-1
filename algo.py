from typing import Optional

import numpy as np

from problem import Problem

class Algorithm:
    def __init__(self, pop_size: int, max_iteration:int, initial_population: Optional[np.ndarray]=None):
        self.max_iteration: int = max_iteration
        self.pop_size: int = pop_size
        self.population: np.ndarray
        self.initial_population: Optional[np.ndarray] = initial_population
        self.vals: np.ndarray
        self.best_sol: np.ndarray
        self.best_val: float
    
    def solve(self, problem: Problem):
        self.reset(problem)
        raise NotImplementedError
    
    def reset(self, problem: Problem):
        if self.initial_population is None:
            self.population = problem.get_random_feasible_solution(self.pop_size)
        else:
            self.population = self.initial_population.copy()
        self.vals = np.asanyarray([problem.evaluate(x) for x in self.population])
        best_idx = np.argmin(self.vals)
        self.best_sol = self.population[best_idx]
        self.best_val = self.vals[best_idx]

    def __repr__(self):
        return "algorithm-base"