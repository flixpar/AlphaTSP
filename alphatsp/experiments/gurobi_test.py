import alphatsp.tsp
import alphatsp.solvers.gurobi
import alphatsp.util

def run():
	tsp = alphatsp.tsp.TSP(100, 2)
	sol, sol_length = alphatsp.solvers.gurobi.exact_gurobi(tsp)
	print(f"Exact solution:\t{sol_length},\t{sol}")
