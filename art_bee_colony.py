import random
import numpy as np

from algo import Algorithm
from problem import Problem

class ArtificialBeeColony:
    def __init__(self, 
                 problem: Problem,
                 num_food_sources: int,
                 max_iterations: int,
                 init_food_sources: np.ndarray, 
                 limit:int=10):
        """
        Initialize the ABC algorithm.

        Parameters:
            objective_function (callable): The objective function to minimize.
            num_food_sources (int): Number of food sources (candidate solutions).
            max_iterations (int): Maximum number of iterations.
            limit (int): Abandonment limit for scout bees.
        """
        self.problem = problem
        self.num_food_sources = num_food_sources
        self.max_iterations = max_iterations
        self.limit = limit
        
        # Initialize food sources (candidate solutions)
        self.food_sources:np.ndarray = init_food_sources
        self.fitness_values:np.ndarray = np.zeros([self.num_food_sources, ], dtype=float)
        self.trials:np.ndarray = np.zeros(self.num_food_sources, dtype=int)
        self.best_solution: np.ndarray = None
        self.best_fitness:float = float("inf")

    def fitness(self, f: float)->float:
        """
        Calculate the fitness of a solution.

        Parameters:
            f (float): Objective function value.

        Returns:
            float: Fitness value.
        """
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)

    def evaluate_fitness(self):
        """
        Evaluate the fitness of all food sources.
        negative the f, because this algo assumes maximization?
        """
        for i in range(self.num_food_sources):
            f = -self.problem.evaluate(self.food_sources[i])
            self.fitness_values[i] = self.fitness(f)
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.food_sources[i]

    def get_new_solution_employed_bee(self, i: int)->np.ndarray:
        new_solution: np.ndarray = self.food_sources[i].copy()
        # Select a random neighbor
        j = random.randint(0, self.num_food_sources-1)
        while i==j:
            j = random.randint(0, self.num_food_sources-1)
        # Select a random dimension
        k = random.randint(0, self.problem.n_var-1)
        r = random.random()
        
        # Generate a new solution
        new_solution[k] = self.food_sources[i,k] + r*(self.food_sources[i,k]-self.food_sources[j,k])    
        
        # Clip the solution to stay within bounds
        new_solution = self.problem.clip_to_bound(new_solution)
        return new_solution

    def employed_bee_phase(self):
        """
        Employed bee phase: Explore new solutions around existing food sources.
        """
        for i in range(self.num_food_sources):
            new_solution: np.ndarray = self.get_new_solution_employed_bee(i)
            # Evaluate the new solution
            new_val = -self.problem.evaluate(new_solution)
            new_fitness = self.fitness(new_val)
            if new_fitness < self.fitness_values[i]:
                self.trials[i]=0
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
            else:
                self.trials[i]+=1

    def onlooker_bee_phase(self):
        """
        Onlooker bee phase: Select food sources based on fitness and explore around them.
        """
        probabilities = self.fitness_values / np.sum(self.fitness_values)
        for i in range(self.num_food_sources):
            if random.random() >= probabilities[i]:
                continue
            # Perform the same steps as the employed bee phase
            new_solution: np.ndarray = self.get_new_solution_employed_bee(i)
            # Evaluate the new solution
            new_val = -self.problem.evaluate(new_solution)
            new_fitness = self.fitness(new_val)
            if new_fitness < self.fitness_values[i]:
                self.trials[i]=0
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
            else:
                self.trials[i]+=1

    def scout_bee_phase(self):
        """
        Scout bee phase: Replace abandoned food sources with new random solutions.
        """
        renew_idxs = np.nonzero(self.trials)[0]
        if len(renew_idxs)==0:
            return
        self.trials[renew_idxs]=0
        self.food_sources[renew_idxs] = self.problem.get_random_feasible_solution(len(renew_idxs))

    def optimize(self):
        """
        Run the ABC optimization algorithm.

        Returns:
            tuple: Best solution and best fitness value.
        """
        self.evaluate_fitness()  # Evaluate initial fitness
        for iteration in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self.evaluate_fitness()  # Update best solution
            
            # print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")
            
            
class ArtificialBeeColonyAlgorithm(Algorithm):
    def __init__(self, 
                 pop_size, 
                 max_iteration):
        super().__init__(pop_size, max_iteration)
        
    def solve(self, problem: Problem):
        init_food_sources: np.ndarray = problem.get_random_feasible_solution(self.pop_size)
        abc: ArtificialBeeColony = ArtificialBeeColony(problem, 
                                                       self.pop_size,
                                                       self.max_iteration, 
                                                       init_food_sources)
        abc.optimize()
        self.population = abc.food_sources
        self.vals = np.asanyarray([problem.evaluate(x) for x in self.population], dtype=float)
        best_idx = np.argsort(self.vals)[0]
        self.best_sol = abc.best_solution
        self.best_val = self.vals[best_idx]
        
    def __repr__(self):
        return "ABC"