import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.util

def run(args):

	tsp = alphatsp.tsp.TSP(args.N, args.D)

	nearest_tour, nearest_tour_length = alphatsp.solvers.heuristics.nearest_insertion(tsp)
	farthest_tour, farthest_tour_length = alphatsp.solvers.heuristics.farthest_insertion(tsp)

	print(f"Nearest:  {nearest_tour_length:.3f} {nearest_tour}")
	print(f"Farthest: {farthest_tour_length:.3f} {farthest_tour}")

	alphatsp.util.display_tour(tsp, nearest_tour, title="Nearest Insertion")
	alphatsp.util.display_tour(tsp, farthest_tour, title="Farthest Insertion")
