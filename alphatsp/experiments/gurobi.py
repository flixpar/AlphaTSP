import alphatsp.tsp
import alphatsp.solvers.gurobi
import alphatsp.util

def run(args):
	tsp = alphatsp.tsp.TSP(args.N, args.D)
	sol, sol_length = alphatsp.solvers.gurobi.exact_gurobi(tsp)
	print(f"Exact solution:\t{sol_length},\t{sol}")
