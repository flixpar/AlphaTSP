import alphatsp.tsp
import alphatsp.solvers.heuristics
import alphatsp.solvers.exact
import alphatsp.solvers.mcts
import alphatsp.util

import alphatsp.solvers.policy_solvers
import alphatsp.solvers.example_generators
import alphatsp.solvers.policy_networks

import torch
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import copy
from multiprocessing import Process, Manager, Lock, Pool

def run(args):

	# setup
	N, D = args.N, args.D
	n_examples = args.n_train_examples
	n_threads = args.n_threads
	n_test_iter = args.n_test_examples
	policy_network = alphatsp.util.get_policy_network(args.policy_network)

	# generate examples
	print("Generating examples and training...")

	manager = Manager()
	train_queue = manager.Queue(100)
	model_queue = manager.Queue()

	finished_lock = manager.Lock()
	finished_lock.acquire(blocking=True)

	for _ in range(n_threads+1):
		model_queue.put(policy_network)

	locks = []
	producers = []
	for _ in range(n_threads):
		l = manager.Lock()
		l.acquire(blocking=True)
		producers.append(Process(target=generate_examples, args=(train_queue, model_queue, n_examples//n_threads, N, D, l, args)))
		locks.append(l)

	c = Process(target=trainer, args=(train_queue, model_queue, locks, finished_lock, args))

	for p in producers:
		p.start()
	c.start()

	for p in producers:
		p.join()

	finished_lock.release()
	c.join()

	train_losses = None
	while not isinstance(train_losses, list):
		train_losses = model_queue.get()
	policy_network = model_queue.get()

	# display training loss
	plt.scatter(x=np.arange(len(train_losses)), y=train_losses, marker='.')
	plt.title("Loss")
	plt.xlabel("examples")
	plt.ylabel("loss")
	plt.savefig("saves/loss_parallel.png")

	# test policy network vs other solvers
	evaluate(policy_network, args)

	# save network
	torch.save(policy_network.state_dict(), "saves/policy_network.pth")

def generate_examples(train_queue, model_queue, n_examples, N, D, l, args):
	policy_network = model_queue.get()
	for i in range(n_examples):
		if not model_queue.empty() and l.acquire(blocking=False):
			policy_network = model_queue.get()
		tsp = alphatsp.tsp.TSP(N, D)
		solver = alphatsp.solvers.example_generators.SelfPlayExampleGenerator(args, tsp, train_queue, policy_network)
		solver.solve()
	if not l.acquire(blocking=False):
		l.release()

def trainer(train_queue, model_queue, locks, finished_lock, args):
	policy_network = model_queue.get()
	trainer = alphatsp.solvers.policy_networks.PolicyNetworkTrainer(policy_network, train_queue)
	it = trainer.n_examples_used
	while True:
		if not train_queue.empty():
			trainer.train_example()
			if trainer.n_examples_used//1000 > it//1000:
				for i in range(len(locks)):
					model_queue.put(copy.deepcopy(policy_network))
					if not locks[i].acquire(blocking=False):
						locks[i].release()
			if trainer.n_examples_used//1000 > it//1000:
				trainer.save_model()
				evaluate(policy_network, args)
			it = trainer.n_examples_used
		elif finished_lock.acquire(blocking=False):
			model_queue.put(trainer.losses)
			model_queue.put(policy_network)
			for l in locks:
				if not l.acquire(blocking=False):
					l.release()
			return 0

def evaluate(policy_network, args):
	n_test_iter, N, D = args.n_test_examples, args.N, args.D

	print("Testing...")
	results = []
	
	with Pool(args.n_threads) as pool:
		result_handlers = [pool.apply_async(evaluate_single, (policy_network, N, D, args)) for _ in range(n_test_iter)]
		for handle in result_handlers:
			results.append(handle.get())

	policy_lens, policymcts_lens, mcts_lens, greedy_lens, exact_lens = zip(*results)

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

def evaluate_single(policy_network, N, D, args):
	tsp = alphatsp.tsp.TSP(N, D)

	# policy only
	policy_solver = alphatsp.solvers.policy_solvers.PolicySolver(args, tsp, policy_network)
	_, policy_tour_len = policy_solver.solve()

	# policy + mcts
	policymcts_solver = alphatsp.solvers.policy_solvers.PolicyMCTSSolver(args, tsp, policy_network)
	_, policymcts_tour_len = policymcts_solver.solve()

	# mcts
	mcts_solver = alphatsp.solvers.mcts.MCTSSolver(args, tsp)
	_, mcts_tour_len = mcts_solver.solve()

	# benchmarks
	_, greedy_tour_len = alphatsp.solvers.heuristics.nearest_greedy(tsp)
	_, exact_tour_len = alphatsp.solvers.exact.exact(tsp)

	return (policy_tour_len, policymcts_tour_len, mcts_tour_len, greedy_tour_len, exact_tour_len)
