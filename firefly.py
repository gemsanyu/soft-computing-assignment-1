import random
from typing import List

import numpy as np

from algo import Algorithm
from problem import Problem

class Firefly:
    """Represents a single firefly in the Firefly Algorithm."""
    def __init__(self, problem:Problem, alpha=0.2, beta_0=0.2, gamma=1.0, delta=0.97):
        """
        Initializes a firefly.
        Args|Parameters
            dim (int): Dimension of objective function, calculated from bounds
            bounds (list of tuples): NP Array of (lower_bound, upper_bound) tuples for each dimension.
            alpha (float): Random number parameter.
            beta_0 (float): Minimum attractiveness.
            gamma (float): Absorption coefficient.
            position: Current location
            intensity: Intensity of the light
        """
        self.problem:Problem = problem
        self.alpha:float = alpha
        self.beta_0:float = beta_0
        self.gamma:float = gamma
        self.delta:float = delta
        self.position: np.ndarray = self.problem.get_random_feasible_solution(1).ravel()
        self.intensity: float = self.problem.evaluate(self.position)
        
    def update_position(self, nearby_firefly: "Firefly"):
        """Update position based on distance, clip any over bounds"""
        distance = np.linalg.norm(self.position-nearby_firefly.position)
        r = np.random.random(self.position.shape)
        scale = self.problem.xrange
        new_position: np.ndarray = self.position +\
            self.beta_0*np.exp(-self.gamma*distance**2)*(nearby_firefly.position-self.position) +\
            self.alpha*self.delta*(r-0.5)*scale
        self.position = new_position
        self.intensity = self.problem.evaluate(self.position)

class FireflySwarm:
    
    def __init__(self, 
                 problem: Problem, 
                 num_fireflies: int, 
                 iterations: int):
        self.iterations = iterations
        self.num_fireflies = num_fireflies
        self.fireflies: List[Firefly] = [Firefly(problem) for _ in range(num_fireflies)]    
            
    def optimize(self):
        """Update the position of each firefly to the brighter firefly depending on problem
                            
                    > if minimization problem
                    < if maximization problem
        """
        for t in range(self.iterations):
            for i in range(self.num_fireflies):
                for j in range(i):
                    if self.fireflies[i].intensity > self.fireflies[j].intensity:
                        self.fireflies[i].update_position(self.fireflies[j])

class FireflyOptimizationAlgorithm(Algorithm):
    def __init__(self, pop_size, max_iteration):
        super().__init__(pop_size, max_iteration)
        
    def solve(self, problem:Problem):
        firefly_swarm: FireflySwarm = FireflySwarm(problem, self.pop_size, self.max_iteration)
        firefly_swarm.optimize()
        self.population = np.stack([firefly.position for firefly in firefly_swarm.fireflies])
        self.vals = np.asanyarray([problem.evaluate(x) for x in self.population], dtype=float)
        best_idx = np.argsort(self.vals)[0]
        self.best_sol = self.population[best_idx]
        self.best_val = self.vals[best_idx]
        
    def __repr__(self):
        return "FFA"