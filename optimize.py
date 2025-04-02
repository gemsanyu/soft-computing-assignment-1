from dataclasses import dataclass
import time

import numpy as np

from algo import Algorithm
from problem import Problem

@dataclass
class OptimizationResult:
    final_population: np.ndarray
    final_population_val: np.ndarray
    opt_sol: np.ndarray
    opt_val: float
    runtime: float

def minimize(problem: Problem, algo: Algorithm)->OptimizationResult:
    start = time.time()
    algo.solve(problem)
    end = time.time()
    runtime = end-start
    return OptimizationResult(
        algo.population,
        algo.vals,
        algo.best_sol,
        algo.best_val,
        runtime
    )