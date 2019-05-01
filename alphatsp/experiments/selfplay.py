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

import copy
from torch.multiprocessing import Process, Manager, Lock
torch.multiprocessing.set_start_method("spawn", force="True")

def run():

	# setup
	N, D = 30, 2
	n_examples = 10000
	n_threads = 8
	n_test_iter = 20
	policy_network = alphatsp.solvers.policy.PolicyNetwork()

	# generate examples
	print("Generating examples and training...")

	manager = Manager()
	train_queue = manager.Queue()
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
		producers.append(Process(target=generate_examples, args=(train_queue, model_queue, n_examples//n_threads, N, D, l)))
		locks.append(l)

	c = Process(target=trainer, args=(train_queue, model_queue, locks, finished_lock))

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
	evaluate(policy_network, n_test_iter, N, D)

	# save network
	torch.save(policy_network.state_dict(), "saves/policy_network.pth")

def generate_examples(train_queue, model_queue, n_examples, N, D, l):
	policy_network = model_queue.get()
	for i in range(n_examples):
		if not model_queue.empty() and l.acquire(blocking=False):
			policy_network = model_queue.get()
		tsp = alphatsp.tsp.TSP(N, D)
		solver = alphatsp.solvers.policy.SelfPlayExampleGenerator(tsp, train_queue, policy_network)
		solver.solve()
	if not l.acquire(blocking=False):
		l.release()

def trainer(train_queue, model_queue, locks, finished_lock):
	policy_network = model_queue.get()
	trainer = alphatsp.solvers.policy.PolicyNetworkTrainer(policy_network, train_queue)
	it = trainer.n_examples_used
	while True:
		if not train_queue.empty():
			trainer.train_all()
			if trainer.n_examples_used//1000 > it//1000:
				for i in range(len(locks)):
					model_queue.put(copy.deepcopy(policy_network))
					if not locks[i].acquire(blocking=False):
						locks[i].release()
			if trainer.n_examples_used//10000 > it//10000:
				trainer.save_model()
			it = trainer.n_examples_used
		elif finished_lock.acquire(blocking=False):
			model_queue.put(trainer.losses)
			model_queue.put(policy_network)
			for l in locks:
				if not l.acquire(blocking=False):
					l.release()
			return 0

def evaluate(policy_network, n_test_iter, N, D):

	print("Testing...")
	policy_lens, policymcts_lens, mcts_lens, greedy_lens, exact_lens = [], [], [], [], []
	for _ in range(n_test_iter):

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
