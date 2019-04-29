import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.exact
import alphatsp.solvers.policy
import alphatsp.solvers.mcts
import alphatsp.util

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import tqdm
import queue

def run():

	# setup
	N, D = 30, 2
	n_examples = 500
	policy_network = alphatsp.solvers.policy.PolicyNetwork()

	# generate examples
	print("Generating examples...")
	train_queue = queue.Queue()
	for _ in tqdm.tqdm(range(n_examples)):
		tsp = alphatsp.tsp.TSP(N, D)
		solver = alphatsp.solvers.policy.MCTSExampleGenerator(tsp, train_queue)
		solver.solve()

	# train policy network
	print("Training...")
	trainer = alphatsp.solvers.policy.PolicyNetworkTrainer(policy_network, train_queue)
	trainer.train_all()
	policy_network = trainer.model

	# display training loss
	plt.scatter(x=np.arange(len(trainer.losses)), y=trainer.losses, marker='.')
	plt.title("Loss")
	plt.xlabel("examples")
	plt.ylabel("loss")
	plt.savefig("saves/loss.png")

	# test policy network vs other solvers
	print("Testing...")
	policy_lens, policymcts_lens, mcts_lens, greedy_lens, exact_lens = [], [], [], [], []
	for _ in range(20):

		tsp = alphatsp.tsp.TSP(N, D)

		# policy only
		policy_solver = alphatsp.solvers.policy.PolicySolver(tsp, policy_network)
		policy_tour, policy_tour_len = policy_solver.solve()

		# policy + mcts
		policymcts_solver = alphatsp.solvers.policy.PolicyMCTSSolver(tsp, policy_network)
		policymcts_tour, policymcts_tour_len = policymcts_solver.solve()

		# mcts
		mcts_solver = alphatsp.solvers.mcts.MCTSSolver(tsp)
		mcts_tour, mcts_tour_len = mcts_solver.solve()

		# benchmarks
		greedy_tour, greedy_tour_len = alphatsp.solvers.heuristics.nearest_greedy(tsp)
		exact_tour, exact_tour_len = alphatsp.solvers.exact.exact(tsp)

		# log lengths
		policy_lens.append(policy_tour_len)
		policymcts_lens.append(policymcts_tour_len)
		mcts_lens.append(mcts_tour_len)
		greedy_lens.append(greedy_tour_len)
		exact_lens.append(exact_tour_len)

	# average results
	policy_avg     = np.mean(policy_lens)
	policymcts_avg = np.mean(policymcts_lens)
	mcts_avg       = np.mean(mcts_lens)
	greedy_avg     = np.mean(greedy_lens)
	exact_avg      = np.mean(exact_lens)

	# print results
	print("\nResults:")
	print(f"Policy:\t\t{policy_avg}")
	print(f"Policy+MCTS:\t{policymcts_avg}")
	print(f"MCTS:\t\t{mcts_avg}")
	print(f"Greedy:\t\t{greedy_avg}")
	print(f"Exact:\t\t{exact_avg}")

	# save network
	torch.save(policy_network.state_dict(), "saves/policy_network.pth")
