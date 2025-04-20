import math
import random
from typing import Optional, List
import multiprocessing as mp

import numpy as np
from scipy.stats import mannwhitneyu

from algo import Algorithm
from problem import Problem, decompose_problem_space
from pso import PSO
from art_bee_colony import ArtificialBeeColonyAlgorithm
from firefly import FireflyOptimizationAlgorithm


class Island:
    def __init__(self, algo: Algorithm, problem: Problem):
        self.algo: Algorithm = algo
        self.problem: Problem = problem    
        
def update_island(island:Island):
    for i in range(100):
        island.algo.update()
    return island

   

def setup_island(algo_name:str, 
                 problem:Problem, 
                 max_iteration,
                 pop_size,
                 current_iteration,
                 initial_population=None)->Island:
    if algo_name == "PSO":
        algo= PSO(pop_size, max_iteration, -1e-3, 1e-3, 0.4, 1.2, initial_population=initial_population, current_iteration=current_iteration)
    elif algo_name == "ABC":
        algo= ArtificialBeeColonyAlgorithm(pop_size, max_iteration)
    elif algo_name == "FFA":
        algo= FireflyOptimizationAlgorithm(pop_size, max_iteration, current_iteration, initial_population)
    return Island(algo, problem)

class EvoArena(Algorithm):
    def __init__(self, 
                 pop_size: int,
                 max_iteration: int,
                 num_islands:int=10, 
                 tournament_interval: int=100,
                 current_iteration: int=1):
        super().__init__(pop_size, max_iteration, None)
        self.max_iteration: int = max_iteration
        self.current_iteration = current_iteration        
        self.problem: Problem
        self.num_islands = num_islands
        self.islands: List[Island]
        self.algo_names = ["PSO","FFA","ABC"]
        self.tournament_interval = tournament_interval
        self.pool = mp.Pool(6)
        

    def set_problem(self, problem: Problem, reset=False):
        self.problem = problem
        if reset:
            self.reset(problem)
            problems = decompose_problem_space(self.problem, self.num_islands)
            self.islands = []
            for i in range(self.num_islands):
                algo_name = random.choice(self.algo_names)
                self.islands.append(setup_island(algo_name, problems[i], self.max_iteration, self.pop_size, current_iteration=1))
                self.islands[i].algo.set_problem(problems[i], reset=True)
                
    def update(self):
        self.islands = self.pool.map(update_island, self.islands)

        
        for i, island in enumerate(self.islands):
            # island.algo.update()
            if self.best_val > island.algo.best_val:
                self.best_sol = island.algo.best_sol
                self.best_val = island.algo.best_val
        self.current_iteration += self.tournament_interval

        # if self.current_iteration % self.tournament_interval != 0:
        #     return

        # Tournament
        random_idxs = np.arange(self.num_islands)
        np.random.shuffle(random_idxs)
        matches = [(random_idxs[i],random_idxs[i+1]) for i in range(0, self.num_islands, 2)]
        for (i1, i2) in matches:
            vals1 = self.islands[i1].algo.vals
            vals2 = self.islands[i2].algo.vals
            clear_winner_exists = False
            stat, p_value = mannwhitneyu(vals1, vals2, alternative="less")  # is i1 < i2?
            if p_value < 0.05:  # statistically significant
                winner, loser = i1, i2
                clear_winner_exists = True
            else:
                # Test reverse
                stat, p_value = mannwhitneyu(vals2, vals1, alternative="less")
                if p_value < 0.05:
                    winner, loser = i2, i1
                    clear_winner_exists = True
        
            if clear_winner_exists:            
                winner_algo_name = str(self.islands[winner].algo)
                winner_problem = self.islands[winner].problem
                new_problems = decompose_problem_space(winner_problem, 2)
                self.islands[winner].problem = new_problems[0]
                self.islands[winner].algo.set_problem(new_problems[0])
                
                new_island = setup_island(winner_algo_name, 
                                          new_problems[1], 
                                          self.max_iteration,
                                          self.pop_size,
                                          self.current_iteration)
                new_island.algo.set_problem(new_problems[1], reset=True)
                self.islands[loser] = new_island
            else:
                # randomize one of the algo
                chosen = i1 if random.random()<=0.5 else i2
                chosen_algo_name = random.choice(self.algo_names)
                problem = self.islands[chosen].problem
                new_island = setup_island(chosen_algo_name, problem, self.max_iteration, self.pop_size, self.current_iteration)
                new_island.algo.set_problem(problem, reset=True)
                self.islands[chosen] = new_island
            
    
    def solve(self, problem):
        self.set_problem(problem, reset=True)
        for t in range(self.max_iteration//self.tournament_interval):
            self.update()
    
    def __repr__(self):
        return "EvoArena"