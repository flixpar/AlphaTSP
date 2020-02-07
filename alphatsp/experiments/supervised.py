import alphatsp.tsp
import alphatsp.util

import alphatsp.solvers.policy_solvers
from alphatsp.solvers.example_generators import NNExampleGenerator
from alphatsp.solvers.policy_networks import SupervisedPolicyNetworkTrainer

import torch
import numpy as np

import copy

from torch.multiprocessing import Process, Manager


def run(args):

	# setup
	N, D = args.N, args.D
	n_examples = args.n_train_examples
	n_threads = args.n_threads
	n_examples_per_thread = n_examples//n_threads

	# create policy network
	policy_network = alphatsp.util.get_policy_network(args.policy_network)

	# generate examples
	print("Generating examples and training...")

	manager = Manager()
	train_queue = manager.Queue()
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
		print("Experiment failed.")
		return -1

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
