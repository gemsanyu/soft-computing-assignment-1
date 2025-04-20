import random
import numpy as np
from typing import Optional

from algo import Algorithm
from problem import Problem


@np.vectorize
def compute_fitness(f: float)->float:
    return 1 / (1 + f) if f >= 0 else 1 + abs(f)

class ArtificialBeeColonyAlgorithm(Algorithm):
    def __init__(self, 
                 pop_size, 
                 max_iteration,
                 limit=10,
                 initial_population: Optional[np.ndarray]=None,
                 current_iteration: int=1):
        super().__init__(pop_size, max_iteration, initial_population)
        self.current_iteration = current_iteration
        self.fitness_values:np.ndarray 
        self.trials:np.ndarray 
        self.best_fitness: float
        self.sol_with_best_fitness: np.ndarray
        self.limit = limit
        
    
    def set_problem(self, problem: Problem, reset=False):
        self.problem = problem
        if reset:
            self.reset(problem)
            self.fitness_values = np.zeros([self.pop_size, ], dtype=float)
            self.trials = np.zeros(self.pop_size, dtype=int)
            self.best_fitness = 0
            self.sol_with_best_fitness = None
        else:
            self.population = np.stack([problem.clip_to_bound(x) for x in self.population])
            self.vals = np.asanyarray([problem.evaluate(x) for x in self.population])
            self.evaluate_fitness()
    
    def evaluate_fitness(self):
        self.fitness_values = compute_fitness(-self.vals)
        best_idx = np.argmin(self.fitness_values)
        self.best_fitness = self.fitness_values[best_idx]
        self.sol_with_best_fitness = self.population[best_idx]
    
    def update(self):
        self.current_iteration += 1
        self.evaluate_fitness()
        self.employed_bee_phase()
        self.onlooker_bee_phase()
        self.scout_bee_phase()
        self.evaluate_fitness()
        best_idx = np.argmin(self.vals)
        self.best_sol = self.population[best_idx]
        self.best_val = self.vals[best_idx]

    def get_new_solution_employed_bee(self, i: int)->np.ndarray:
        new_solution: np.ndarray = self.population[i].copy()
        # Select a random neighbor
        j = random.randint(0, self.pop_size-1)
        while i==j:
            j = random.randint(0, self.pop_size-1)
        # Select a random dimension
        k = random.randint(0, self.problem.n_var-1)
        r = random.random()
        
        # Generate a new solution
        new_solution[k] = self.population[i,k] + r*(self.population[i,k]-self.population[j,k])    
        
        # Clip the solution to stay within bounds
        new_solution = self.problem.clip_to_bound(new_solution)
        return new_solution
    
    def employed_bee_phase(self):
        for i in range(self.pop_size):
            new_solution: np.ndarray = self.get_new_solution_employed_bee(i)
            # Evaluate the new solution
            new_val = -self.problem.evaluate(new_solution)
            new_fitness = compute_fitness(new_val)
            if new_fitness < self.fitness_values[i]:
                self.trials[i]=0
                self.population[i] = new_solution
                self.fitness_values[i] = new_fitness
            else:
                self.trials[i]+=1
                
    def onlooker_bee_phase(self):
        probabilities = self.fitness_values / np.sum(self.fitness_values)
        for i in range(self.pop_size):
            if random.random() >= probabilities[i]:
                continue
            # Perform the same steps as the employed bee phase
            new_solution: np.ndarray = self.get_new_solution_employed_bee(i)
            # Evaluate the new solution
            new_val = -self.problem.evaluate(new_solution)
            new_fitness = compute_fitness(new_val)
            if new_fitness < self.fitness_values[i]:
                self.trials[i]=0
                self.population[i] = new_solution
                self.fitness_values[i] = new_fitness
            else:
                self.trials[i]+=1
                
    def scout_bee_phase(self):
        renew_idxs = np.nonzero(self.trials)[0]
        if len(renew_idxs)==0:
            return
        self.trials[renew_idxs]=0
        self.population[renew_idxs] = self.problem.get_random_feasible_solution(len(renew_idxs))
                 
    def solve(self, problem: Problem):
        self.set_problem(problem, reset=True)
        for t in range(self.max_iteration):
            self.update()
        
    def __repr__(self):
        return "ABC"