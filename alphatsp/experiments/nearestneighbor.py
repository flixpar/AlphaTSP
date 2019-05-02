import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.util

def run(args):
	tsp = alphatsp.tsp.TSP(args.N, args.D)
	sol, sol_length = alphatsp.solvers.heuristics.nearest_greedy(tsp)
	print(f"Tour length: {sol_length}")
	print(f"Tour: {sol}")
	alphatsp.util.display_tour(tsp, sol)
