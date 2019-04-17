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
	node = alphatsp.solvers.mcts.MCTSNode(n=n)
	while not node.is_leaf():
		node = alphatsp.solvers.mcts.mcts(node, tsp, 1000)
	mcts_tour = node.get_tour()
	mcts_payoff = tsp.tour_length(mcts_tour)

	print(f"MCTS Tour:\t{mcts_payoff},\t{mcts_tour}")

	# Nearest neighbor greedy solver
	tsp = alphatsp.tsp.TSP(n, d, points=points)
	greedy_sol, greedy_sol_length = alphatsp.solvers.heuristics.nearest_greedy(tsp)

	print(f"Greedy Tour:\t{greedy_sol_length},\t{greedy_sol}")
