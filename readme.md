🧠 Evolutionary Optimization Experiments

This project implements and compares several nature-inspired optimization algorithms:

- Particle Swarm Optimization (PSO)
- Firefly Algorithm (FFA)
- Artificial Bee Colony (ABC)
- EvoArena (a multi-algorithm tournament-style environment)

These algorithms are tested on benchmark optimization problems such as Rastrigin and Styblinski–Tang.

---------------------
How to Run:

    python main.py --algo_name <algo> --problem_name <problem> --n_var <num_vars> --pop_size <size> --max_iteration <max_iter>

Required Arguments:

    --algo_name        Optimization algorithm: pso, ffa, abc, evoarena
    --problem_name     Benchmark function: rastrigin, styblinski
    --n_var            Number of decision variables
    --pop_size         Population size
    --max_iteration    Maximum number of iterations


---------------------
Output:

Results are saved to result.csv in the format:

    Algorithm,Problem,N_Var,Population_Size,Max_Iteration,Opt_Val,Runtime

---------------------
Example Usages:

Run PSO on Rastrigin:

    python main.py --algo_name pso --problem_name rastrigin --n_var 30 --pop_size 50 --max_iteration 300

Run ABC on Styblinski–Tang:

    python main.py --algo_name abc --problem_name styblinski --n_var 20 --pop_size 30 --max_iteration 500

Run EvoArena with default tournament settings:

    python main.py --algo_name evoarena --problem_name rastrigin --n_var 50 --pop_size 60 --max_iteration 500

---------------------
Dependencies:

- Python 3.7+
- numpy