import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

import queue
import numpy as np

from alphatsp.solvers.mcts import MCTSNode


class PolicyNetwork(nn.Module):
    def __init__(self, d=2):
        super(PolicyNetwork, self).__init__()
        self.conv1 = GCNConv(d,  16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 1)
        self.fc    = nn.Linear(16, 1)
    
    def forward(self, graph):
        x, edges, choices = graph.pos, graph.edge_index, graph.y
        
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = self.conv2(x, edges)
        x = F.relu(x)
        
        c = self.conv3(x, edges)
        choice = torch.masked_select(c.squeeze(), choices)
        choice = F.softmax(choice, dim=0)
        
        v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long))
        value = self.fc(v)

        return choice, value

class PolicySolver:

	def __init__(self, tsp, model):
		self.tsp = tsp
		self.root_node = MCTSNode(tsp=self.tsp)
		self.model = model

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = node.best_remaining_policy(model=self.model)
		mcts_tour = node.get_tour()
		mcts_payoff = self.tsp.tour_length(mcts_tour)
		return mcts_tour, mcts_payoff

class PolicyMCTSSolver:

	def __init__(self, tsp, model, iterations=50):
		self.tsp = tsp
		self.root_node = MCTSNode(tsp=self.tsp)
		self.iterations = iterations
		self.model = model

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = self.mcts_search(node)
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
				node = node.select_child_policy(self.model)
		return node

class MCTSExampleGenerator:

	def __init__(self, tsp, example_queue, iterations=1000):
		self.tsp = tsp
		self.root_node = MCTSNode(tsp=self.tsp)
		self.example_queue = example_queue
		self.iterations = iterations

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = self.mcts_search(node)
			self.generate_example(node.parent)
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

	def generate_example(self, node):

		# construct graph
		graph = node.construct_graph()

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
		self.example_queue.put(example)

class MCTSPolicyExampleGenerator:

	def __init__(self, tsp, example_queue, model, iterations=500):
		self.tsp = tsp
		self.root_node = MCTSNode(tsp=self.tsp)
		self.example_queue = example_queue
		self.model = model
		self.iterations = iterations

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = self.mcts_search(node)
			self.generate_example(node.parent)
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
				node = node.select_child_policy(self.model)
		return node

	def generate_example(self, node):

		# construct graph
		graph = node.construct_graph()

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
		self.example_queue.put(example)

class PolicyNetworkTrainer:

	def __init__(self, model, example_queue):

		self.model = model
		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=3e-4)

		self.example_queue = example_queue
		self.losses = []

	def train_example(self):
		self.model.train()

		example = self.example_queue.get()
		graph, choice_probs, value = example["graph"], example["choice_probs"], example["pred_value"]

		pred_choices, pred_value = self.model(graph)
		loss = self.loss_fn(pred_choices, choice_probs) + (0.2 * self.loss_fn(pred_value, value))

		self.losses.append(loss.item())

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def train_all(self):
		self.model.train()
		while not self.example_queue.empty():

			example = self.example_queue.get()
			graph, choice_probs, value = example["graph"], example["choice_probs"], example["pred_value"]

			pred_choices, pred_value = self.model(graph)
			loss = self.loss_fn(pred_choices, choice_probs) + (0.2 * self.loss_fn(pred_value, value))

			self.losses.append(loss.item())

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()