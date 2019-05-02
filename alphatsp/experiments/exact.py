import alphatsp.tsp
import alphatsp.solvers.exact
import alphatsp.util

def run(args):
	tsp = alphatsp.tsp.TSP(args.N, args.D)
	sol, sol_length = alphatsp.solvers.exact.exact(tsp)
	print(f"Exact solution:\t{sol_length},\t{sol}")
