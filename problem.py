import math

import numpy as np
import numba as nb

class Problem:
    def __init__(self, n_var: int, xl: np.ndarray, xu: np.ndarray):
        self.n_var = n_var
        self.opt_sol: np.ndarray
        self.opt_val: float
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.xrange: np.ndarray = xu-xl
    
    def evaluate(self, x: np.ndarray)->float:
        raise NotImplementedError
    
    def clip_to_bound(self, x: np.ndarray)-> np.ndarray:
        return np.clip(x, self.xl, self.xu)
    
    def get_random_feasible_solution(self, num_solutions: int)->np.ndarray:
        return np.random.random([num_solutions, self.n_var])*self.xrange + self.xl
    

@nb.njit(nb.float64(nb.float64[:]), fastmath=True, cache=True)
def r(x: np.ndarray)->float:
    return np.sum(x**2-10*np.cos(2*np.pi*x))
    
class Rastrigin(Problem):
    def __init__(self, n_var:int):
        xl = np.full([n_var,], -5.12, dtype=float)
        xu = np.full([n_var,], 5.12, dtype=float)
        super().__init__(n_var, xl, xu)
        self.opt_sol: np.ndarray = np.zeros([n_var,], dtype=float)
        self.opt_val: float = 0
    
    def evaluate(self, x: np.ndarray)->float:
        return 10*self.n_var + r(x)
        
    def __repr__(self):
        return "Rastrigin"

@nb.njit(nb.float64(nb.float64[:]), fastmath=True, cache=True)
def st(x: np.ndarray)->float:
    return 0.5 * np.sum((x**4) - 16*(x**2) + 5*x)
class StyblinksiTang(Problem):
    def __init__(self, n_var:int):
        xl = np.full([n_var,], -5, dtype=float)
        xu = np.full([n_var,], 5, dtype=float)
        super().__init__(n_var, xl, xu)
        self.opt_sol: np.ndarray = np.full([n_var,], -2.903534, dtype=float)
        self.opt_val: float = n_var*(-39.16599)
    
    def evaluate(self, x: np.ndarray)->float:
        return st(x)
    
    def __repr__(self):
        return "Styblinski-Tang"