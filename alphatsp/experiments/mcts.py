import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.mcts
import alphatsp.solvers.exact
import alphatsp.util

import numpy as np

def run(args):

	n, d = args.N, args.D
	points = np.random.rand(n, d)
	tsp = alphatsp.tsp.TSP(n, d, points=points)

	# MCTS Solver
	mcts_solver = alphatsp.solvers.mcts.MCTSSolver(args, tsp)
	mcts_sol, mcts_sol_length = mcts_solver.solve()
	print(f"MCTS Tour:\t{mcts_sol_length},\t{mcts_sol}")

	# Nearest neighbor greedy solver
	greedy_sol, greedy_sol_length = alphatsp.solvers.heuristics.nearest_greedy(tsp)
	print(f"Greedy Tour:\t{greedy_sol_length},\t{greedy_sol}")

	# Exact solver
	sol, sol_length = alphatsp.solvers.exact.exact(tsp)
	print(f"Exact solution:\t{sol_length},\t{sol}")
