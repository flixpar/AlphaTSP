import alphatsp.tsp
import alphatsp.util

import alphatsp.solvers.policy_solvers
from alphatsp.solvers.example_generators import NNExampleGenerator
from alphatsp.solvers.policy_networks import SupervisedPolicyNetworkTrainer
from alphatsp.logger import Logger

import torch
import numpy as np

import math
import copy

from torch.multiprocessing import Process, Manager


def run(args):

	# setup
	N, D = args.N, args.D
	n_examples = args.n_train_examples
	n_threads = args.n_threads
	n_examples_per_thread = math.ceil(n_examples/n_threads)
	logger = Logger()

	# create policy network
	policy_network = alphatsp.util.get_policy_network(args.policy_network)
	policy_network = policy_network.to(device)

	# generate examples
	logger.print("Generating examples and training...")

	manager = Manager()
	train_queue = manager.Queue(maxsize=5000)
	shared_dict = manager.dict()

	shared_dict["success"] = False

	producers = []
	for _ in range(n_threads):
		producers.append(Process(target=generate_examples, args=(n_examples_per_thread, train_queue, args)))

	for p in producers:
		p.start()

	c = Process(target=train, args=(policy_network, train_queue, shared_dict, args))
	c.start()

	for p in producers:
		p.join()
	train_queue.put(None)

	c.join()

	status = shared_dict["success"]
	if not status:
		logger.print("Experiment failed.")
		return -1
	else: logger.print("Training completed successfully!")

def generate_examples(n_examples, train_queue, args):
	generator = NNExampleGenerator(train_queue, args)
	generator.generate_examples(n_examples)
	return

def train(policy_network, train_queue, shared_dict, args):
	trainer = SupervisedPolicyNetworkTrainer(policy_network, train_queue)
	trainer.train_all()
	shared_dict["model"] = copy.deepcopy(trainer.model.cpu())
	shared_dict["success"] = True
	return

def test(policy_network, args):

	logger.print("Testing...")
	policy_lens, greedy_lens, exact_lens = [], [], []
	for _ in range(args.n_test_examples):

		tsp = alphatsp.tsp.TSP(args.N, args.D)

		# policy only
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
	policy_avg     = np.mean(policy_lens)
	greedy_avg     = np.mean(greedy_lens)
	exact_avg      = np.mean(exact_lens)

	# print results
	logger.print("\nResults:")
	logger.print(f"Policy:\t\t{policy_avg}")
	logger.print(f"Greedy:\t\t{greedy_avg}")
	logger.print(f"Exact:\t\t{exact_avg}")
