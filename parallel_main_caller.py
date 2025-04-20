import itertools
import multiprocessing as mp
import subprocess

def call_main(problem_name: str,
              algo_name: str,
              n_var: int,
              pop_size: int,
              max_iteration: int,
              trial_idx: int):
    subprocess.run(["python", 
                    "main.py",
                    "--algo-name",
                    algo_name,
                    "--problem-name",
                    problem_name,
                    "--n-var",
                    str(n_var),
                    "--pop-size",
                    str(pop_size),
                    "--max-iteration",
                    str(max_iteration)])

if __name__ == "__main__":
    algo_names = [
                  "pso",
                  "abc",
                  "ffa",
                #   "evoarena",
                  ]
    problem_names = [
                    # "rastrigin",
                     "styblinski-tang"]
    n_vars = [2,5,10,30,50]
    pop_sizes = [50]
    max_iterations =[1000]
    trial_idxs = list(range(20))
    args = list(itertools.product(problem_names, algo_names,  n_vars, pop_sizes, max_iterations, trial_idxs))
    for arg in args:
        call_main(*arg)
    # with mp.Pool(6) as pool:
    #     pool.starmap(call_main, args)