import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="experiment arguments.")
    parser.add_argument("--algo-name",
                        type=str,
                        required=True,
                        choices=["pso","abc","ffa","evoarena"],
                        help="algo name")
    parser.add_argument("--problem-name",
                        type=str,
                        required=True,
                        choices=["rastrigin","styblinski-tang"],
                        help="problem name")
    parser.add_argument("--n-var",
                        type=int,
                        required=True,
                        help="number of variable in problem")
    
    parser.add_argument("--pop-size",
                        type=int,
                        required=True,
                        help="population size")
    
    parser.add_argument("--max-iteration",
                        type=int,
                        required=True,
                        help="max iteration number")
    
    return parser.parse_args()
    
    