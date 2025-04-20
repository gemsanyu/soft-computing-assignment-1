import numpy as np

from algo import Algorithm
from problem import Problem


class FireflyOptimizationAlgorithm(Algorithm):
    def __init__(self, pop_size, max_iteration,
                 current_iteration=1, 
                 initial_population=None,
                 alpha=0.2, 
                 beta_0=0.2, 
                 gamma=1.0, 
                 delta=0.97):
        super().__init__(pop_size, max_iteration, initial_population)
        self.alpha:float = alpha
        self.beta_0:float = beta_0
        self.gamma:float = gamma
        self.delta:float = delta
        self.current_iteration = current_iteration
        self.problem: Problem
    
    def set_problem(self, problem: Problem, reset=False):
        self.problem = problem
        if reset:
            self.reset(problem)
        else:
            self.population = np.stack([problem.clip_to_bound(x) for x in self.population])
            self.vals = np.asanyarray([problem.evaluate(x) for x in self.population])
    
    
    def update(self):
        self.current_iteration += 1
        for i in range(self.pop_size):
            for j in range(i):
                if self.vals[i] <= self.vals[j]:
                    continue
                distance = np.linalg.norm(self.population[i]-self.population[j])
                r = np.random.random(self.population[i].shape)
                scale = self.problem.xrange
                new_position: np.ndarray = self.population[i] +\
                    self.beta_0*np.exp(-self.gamma*distance**2)*(self.population[j]-self.population[i]) +\
                    self.alpha*self.delta*(r-0.5)*scale
                new_position = self.problem.clip_to_bound(new_position)
                self.population[i] = new_position
                self.vals[i] = self.problem.evaluate(new_position)
        best_idx = np.argmin(self.vals)
        self.best_sol, self.best_val = self.population[best_idx], self.vals[best_idx]
        
    def solve(self, problem:Problem):
        self.set_problem(problem, reset=True)
        for t in range(self.max_iteration):
            self.update()
        
        
    def __repr__(self):
        return "FFA"