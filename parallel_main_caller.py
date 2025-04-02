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
    import os

    # Set the number of threads (adjust as needed)
    os.environ["OMP_NUM_THREADS"] = "2"  # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "2"  # Intel MKL threads
    os.environ["OPENBLAS_NUM_THREADS"] = "2"  # OpenBLAS threads
    os.environ["NUMEXPR_NUM_THREADS"] = "2"  # NumExpr threads


    algo_names = [
                    # "pso",
                #   "abc",
                  "ffa",
                  ]
    problem_names = ["rastrigin","styblinski-tang"]
    n_vars = [2,4,10,50]
    pop_sizes = [10, 50, 100, 500, 1000]
    max_iterations =[10000]
    trial_idxs = list(range(10))
    args = list(itertools.product(problem_names, algo_names,  n_vars, pop_sizes, max_iterations, trial_idxs))
    with mp.Pool(6) as pool:
        pool.starmap(call_main, args)