import math
from random import random
from typing import List

import numpy as np

from algo import Algorithm
from problem import Problem


class Particle:
    def __init__(self,
                 position: np.ndarray,
                 min_velocity:float,
                 max_velocity:float,
                 init_inertia:float,
                 dt_inertia: float,
                 problem:Problem):
        self.problem: Problem = problem
        self.position: np.ndarray = position  # Current random position of the particle
        self.fitness: float = problem.evaluate(position)
        self.velocity: np.ndarray = np.random.random(position.shape)  # Initial random velocity
        self.p_best: np.ndarray = position.copy()     # Best position found by this particle
        self.p_best_fitness: float = self.fitness   # Fitness of the best position
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.inertia: float = init_inertia
        self.dt_inertia: float = dt_inertia
        

    def update(self, g_best:np.ndarray, c1=0.5, c2=0.5):
        g_best_fitness = self.problem.evaluate(g_best)
        r1, r2 = random(), random()

        # Update velocity
        new_velocity = self.inertia*self.velocity + c1*r1*(self.p_best-self.position) + c2*r2*(g_best-self.position)
        new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)
        self.velocity = new_velocity
        # Limit velocity to avoid explosion using numpy.clip
        
        # Update position
        new_position = self.position + new_velocity
        new_position = self.problem.clip_to_bound(new_position)
        self.position = new_position

        # Constrain position to search space bounds using numpy.clip
        new_fitness = self.problem.evaluate(new_position)
        self.fitness = new_fitness
        if new_fitness < self.p_best_fitness:
            self.p_best, self.p_best_fitness = new_position, new_fitness
            if new_fitness < g_best_fitness:
                g_best, g_best_fitness = new_position, new_fitness

        # update inertia
        self.inertia += self.dt_inertia

        # Update personal best if the current position is better
        return g_best, g_best_fitness

class Swarm:
    def __init__(self,
                 initial_positions: np.ndarray,
                 num_particles:int,
                 min_velocity:float,
                 max_velocity:float,
                 init_inertia:float,
                 dt_inertia:float,
                 problem: Problem):
        self.particles: List[Particle] = [Particle(initial_positions[i], min_velocity, max_velocity, init_inertia, dt_inertia, problem) for i in range(num_particles)]
        self.problem: Problem = problem
        self.g_best: np.ndarray = self.find_global_best()
        self.g_best_fitness: float = problem.evaluate(self.g_best)
        # Initialize particles with random positions

    def find_global_best(self):
        g_best_fitness = 99999999
        g_best = 0
        for particle in self.particles:
            if particle.p_best_fitness<g_best_fitness:
                g_best, g_best_fitness = particle.p_best, particle.p_best_fitness
        return g_best

    def update(self):             
        for particle in self.particles:
            self.g_best, self.g_best_fitness = particle.update(self.g_best)
    

class PSO(Algorithm):
    def __init__(self, 
                 pop_size: int,
                 max_iteration: int, 
                 min_velocity: float, 
                 max_velocity: float,
                 min_inertia: float,
                 max_inertia: float):
        super().__init__(pop_size, max_iteration)
        self.max_iteration: int = max_iteration
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.min_inertia: float = min_inertia
        self.max_inertia: float = max_inertia
        self.init_inertia: float = max_inertia
        self.dt_inertia = (self.min_inertia-self.max_inertia)/self.max_iteration

    def solve(self, problem: Problem):
        self.reset(problem)
        swarm = Swarm(self.population, self.pop_size, self.min_velocity, self.max_velocity, self.init_inertia, self.dt_inertia, problem)
        for t in range(self.max_iteration):
            swarm.update()
    
        # maybe just record the p_best as the final population
        for i in range(self.pop_size):
            self.population[i] = swarm.particles[i].p_best
            self.vals[i] = swarm.particles[i].p_best_fitness
    
    def __repr__(self):
        return "PSO"