import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.mcts
import alphatsp.util

import numpy as np

def run():

	n, d = 30, 2
	points = np.random.rand(n, d)

	# MCTS Solver
	state = alphatsp.solvers.mcts.TSPState(points)
	while state.has_moves():
		m = alphatsp.solvers.mcts.UCT(rootstate = state, itermax = 1000, verbose = False)
		state.do_move(m)
	mcts_payoff = state.get_tour_length()

	print(f"MCTS Tour:\t{mcts_payoff},\t{state.tour}")

	# Nearest neighbor greedy solver
	tsp = alphatsp.tsp.TSP(n, d, points=points)
	greedy_sol, greedy_sol_length = alphatsp.solvers.heuristics.nearest_greedy(tsp)

	print(f"Greedy Tour:\t{greedy_sol_length},\t{greedy_sol}")
