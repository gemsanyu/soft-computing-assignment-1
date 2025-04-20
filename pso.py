import math
from random import random
from typing import Optional

import numpy as np

from algo import Algorithm
from problem import Problem

class PSO(Algorithm):
    def __init__(self, 
                 pop_size: int,
                 max_iteration: int, 
                 min_velocity: float, 
                 max_velocity: float,
                 min_inertia: float,
                 max_inertia: float,
                 c1: float = 0.5,
                 c2: float = 0.5,
                 initial_population: Optional[np.ndarray]=None,
                 current_iteration: int=1):
        super().__init__(pop_size, max_iteration, initial_population)
        self.max_iteration: int = max_iteration
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.min_inertia: float = min_inertia
        self.max_inertia: float = max_inertia
        self.init_inertia: float = max_inertia
        self.dt_inertia = (self.min_inertia-self.max_inertia)/self.max_iteration
        self.current_iteration = current_iteration
        
        self.c1, self.c2 = c1, c2
        self.velocities: np.ndarray  # Initial random velocity of every particle
        self.p_bests: np.ndarray     # Best position found by every particle
        self.p_bests_fitness: np.ndarray   # Fitness of the best position
        self.inertias: np.ndarray
        self.g_best: np.ndarray
        self.g_best_val: np.ndarray
        self.problem: Problem

    def set_problem(self, problem: Problem, reset=False):
        self.problem = problem
        if reset:
            self.reset(problem)
            self.inertias = np.full((self.pop_size,), self.init_inertia)
            self.p_bests = self.population.copy()
            self.p_bests_fitness = self.vals.copy()
            self.g_best = self.best_sol.copy()
            self.g_best_val = self.best_val
            self.velocities = np.random.random(self.population.shape)
            self.velocities = np.clip(self.velocities, self.min_velocity, self.max_velocity)
        else:
            self.population = np.stack([problem.clip_to_bound(x) for x in self.population])
            self.vals = np.asanyarray([problem.evaluate(x) for x in self.population])
            self.p_bests = np.stack([problem.clip_to_bound(x) for x in self.p_bests])
            self.p_bests_fitness = np.asanyarray([problem.evaluate(x) for x in self.p_bests])
            is_replace_pbests = self.vals < self.p_bests_fitness
            self.p_bests[is_replace_pbests] = self.population[is_replace_pbests]
            self.p_bests_fitness[is_replace_pbests] = self.vals[is_replace_pbests]
            best_idx = np.argmin(self.p_bests_fitness)
            self.g_best = self.p_bests[best_idx].copy()
            self.g_best_val = self.p_bests_fitness[best_idx].copy()

    def update(self):
        self.current_iteration += 1
        for i in range(self.pop_size):
            p_best = self.p_bests[i]
            position = self.population[i]
            velocity = self.velocities[i]
            r1, r2 = random(), random()

            # Update velocity
            new_velocity = self.inertias[i]*velocity+ self.c1*r1*(p_best-position) + self.c2*r2*(self.g_best-position)
            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)
            self.velocities[i] = new_velocity
            
            # Update position
            new_position = position + new_velocity
            new_position = self.problem.clip_to_bound(new_position)
            self.population[i] = new_position

            # Constrain position to search space bounds using numpy.clip
            new_fitness = self.problem.evaluate(new_position)
            self.vals[i] = new_fitness
            if new_fitness < self.p_bests_fitness[i]:
                self.p_bests[i], self.p_bests_fitness[i] = new_position, new_fitness
                if new_fitness < self.g_best_val:
                    self.g_best, self.g_best_val = new_position, new_fitness

            # update inertia
            self.inertias[i] += self.dt_inertia
        self.best_sol, self.best_val = self.g_best, self.g_best_val
    
    def solve(self, problem):
        self.set_problem(problem, reset=True)
        for t in range(self.max_iteration):
            self.update()
    
    def __repr__(self):
        return "PSO"