import argparse

from tsp import TSP
import solvers
import util

def main(args):
	if args.experiment == "nearest_greedy":
		nearest_greedy()
	else:
		raise ValueError("Invalid experiment selection.")

def nearest_greedy():
	tsp = TSP(100, 2)
	sol, sol_length = solvers.nearest_greedy(tsp)
	print(f"Tour length: {sol_length}")
	print(f"Tour: {sol}")
	util.display_tour(tsp, sol)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, required=True, help="experiment name")
	args = parser.parse_args()
	main(args)
