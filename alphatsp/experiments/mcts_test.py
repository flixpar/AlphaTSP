import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.mcts
import alphatsp.util

import numpy as np

def run():

	n, d = 30, 2
	points = np.random.rand(n, d)

	# MCTS Solver
	tsp = alphatsp.tsp.TSP(n, d, points=points)
	mcts_sol, mvts_sol_length = alphatsp.solvers.mcts.mcts(tsp)

	print(f"MCTS Tour:\t{mvts_sol_length},\t{mcts_sol}")

	# Nearest neighbor greedy solver
	tsp = alphatsp.tsp.TSP(n, d, points=points)
	greedy_sol, greedy_sol_length = alphatsp.solvers.heuristics.nearest_greedy(tsp)

	print(f"Greedy Tour:\t{greedy_sol_length},\t{greedy_sol}")
