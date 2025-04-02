import filelock
import pathlib


from arguments import parse_args
from problem import Problem, Rastrigin, StyblinksiTang
from algo import Algorithm
from pso import PSO
from art_bee_colony import ArtificialBeeColonyAlgorithm
from optimize import minimize

def setup_problem(problem_name: str, n_var: int)->Problem:
    if problem_name == "rastrigin":
        return Rastrigin(n_var)
    return StyblinksiTang(n_var)

def setup_algo(algo_name:str,
               pop_size:int, 
               max_iteration:int,
               #pso
               min_velocity:float = -1e-3,
               max_velocity:float = 1e-3,
               min_inertia:float = 0.4,
               max_inertia:float = 1.2,
               #abc

               #ffa
               
               )->Algorithm:
    if algo_name == "pso":
        return PSO(pop_size, max_iteration, min_velocity, max_velocity, min_inertia, max_inertia)
    elif algo_name == "abc":
        return ArtificialBeeColonyAlgorithm(pop_size, max_iteration)
    return Algorithm(pop_size, max_iteration)


def run(problem: Problem, algo: Algorithm):
    filename = "result.csv"
    filepath = pathlib.Path()/filename
    result = minimize(problem, algo)
    lock_filename = "results.lock"
    with filelock.FileLock(lock_filename):
        with open(filepath.absolute(),"a+") as result_file:
            # result_file.write("Algorithm,Problem,N_Var,Population_Size,Max_Iteration,Opt_Val,Runtime\n")
            result_string = f"{algo},{problem},{problem.n_var},{algo.pop_size},{algo.max_iteration},{result.opt_val},{result.runtime}\n"
            result_file.write(result_string)
            result_file.flush()


if __name__ == "__main__":
    args = parse_args()
    problem = setup_problem(args.problem_name, args.n_var)
    algo = setup_algo(args.algo_name, args.pop_size, args.max_iteration)
    run(problem, algo)