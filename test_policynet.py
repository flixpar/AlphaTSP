import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.exact
import alphatsp.util

import alphatsp.solvers.policy_solvers
import alphatsp.solvers.example_generators
import alphatsp.solvers.policy_networks

import torch
import numpy as np

import tqdm

import argparse
import importlib

def main(policy_network, args):

	print("Testing...")
	policy_lens, greedy_lens, exact_lens = [], [], []
	for _ in range(args.n_test_examples):

		tsp = alphatsp.tsp.TSP(args.N, args.D)

		# policy
		policy_solver = alphatsp.solvers.policy_solvers.PolicySolver(args, tsp, policy_network)
		policy_tour, policy_tour_len = policy_solver.solve()

		# benchmarks
		greedy_tour, greedy_tour_len = alphatsp.solvers.heuristics.nearest_greedy(tsp)
		exact_tour, exact_tour_len = alphatsp.solvers.exact.exact(tsp)

		# log lengths
		policy_lens.append(policy_tour_len)
		greedy_lens.append(greedy_tour_len)
		exact_lens.append(exact_tour_len)

	# average results
	policy_avg = np.mean(policy_lens)
	greedy_avg = np.mean(greedy_lens)
	exact_avg  = np.mean(exact_lens)

	# print results
	print("\nResults:")
	print(f"Policy:\t\t{policy_avg}")
	print(f"Greedy:\t\t{greedy_avg}")
	print(f"Exact:\t\t{exact_avg}")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--save", type=str, required=True, help="experiment save identifier")
    parser.add_argument("--iteration", required=True, help="save iteration")
	config = parser.parse_args()

    args_module = importlib.import_module(f"saves.{config.save}.args")
    args = args_module.Args()

    net_path = f"./saves/{config.save}/policynet_{config.iteration:07d}.pth"
    net = alphatsp.util.get_policy_network(args.policy_network)
    net.load_state_dict(torch.load(net_path))

	main(net, args)
