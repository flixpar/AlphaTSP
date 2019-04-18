import alphatsp.tsp
import alphatsp.solvers.exact
import alphatsp.util

def run():
	tsp = alphatsp.tsp.TSP(100, 2)
	sol, sol_length = alphatsp.solvers.exact.exact(tsp)
	print(f"Exact solution:\t{sol_length},\t{sol}")
