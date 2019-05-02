import torch
import copy
from alphatsp.solvers.mcts import MCTSNode, MCTSTree

class MCTSExampleGenerator:

	def __init__(self, args, tsp, example_queue):
		self.tsp = tsp
		self.tree = MCTSTree(args, self.tsp)
		self.root_node = self.tree.root_node
		self.example_queue = example_queue
		self.iterations = args.mcts_iters

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = self.mcts_search(node)
			self.enqueue_example(node.parent)
		mcts_tour = node.get_tour()
		mcts_payoff = self.tsp.tour_length(mcts_tour)
		return mcts_tour, mcts_payoff

	def mcts_search(self, start_node):
		for _ in range(self.iterations):
			node = self.tree_policy(start_node)
			pay = node.simulate()
			node.backprop(pay)
		return start_node.best_child_score()

	def tree_policy(self, start_node):
		node = start_node
		while not node.is_leaf():
			if not node.is_fully_expanded():
				return node.expand()
			else:
				node = node.best_child_uct()
		return node

	def enqueue_example(self, node):

		# construct graph
		graph = node.get_graph()

		# construct labels
		choice_probs = [(child.tour[-1], child.visits) for child in node.children]
		choice_probs = sorted(choice_probs, key = lambda c: c[1])
		choice_probs = [c[0] for c in choice_probs]
		choice_probs = torch.tensor(choice_probs).to(dtype=torch.float)
		choice_probs = choice_probs / choice_probs.sum()
		choice = torch.argmax(choice_probs)
		pred_value = torch.tensor(node.avg_score)

		# add to training queue
		example = {
			"graph": graph,
			"choice_probs": choice_probs,
			"choice": choice,
			"pred_value": pred_value,
		}
		self.example_queue.put(copy.deepcopy(example))

class SelfPlayExampleGenerator(MCTSExampleGenerator):

	def __init__(self, args, tsp, example_queue, model):
		super(SelfPlayExampleGenerator, self).__init__(args, tsp, example_queue)
		self.model = model

	def solve(self):

		node = copy.deepcopy(self.root_node)
		while not node.is_leaf():
			node = node.best_remaining_policy(self.model)
		greedy_tour = node.get_tour()
		greedy_payoff = self.tsp.tour_length(greedy_tour)
		self.root_node.thresh = greedy_payoff

		node = self.root_node
		while not node.is_leaf():
			node = self.mcts_search(node)
			self.enqueue_example(node.parent)
		mcts_tour = node.get_tour()
		mcts_payoff = self.tsp.tour_length(mcts_tour)

		return mcts_tour, mcts_payoff
