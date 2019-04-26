import copy
import random
import math

class MCTSNode:

	def __init__(self, p=None, t=[0], r=None, n=-1):
		self.parent = p
		self.tour = t
		self.remaining = r if r is not None else list(range(1, n))
		self.visits = 0
		self.total_score = 0
		self.avg_score = 0
		self.c = 0.7
		self.n = n
		self.children = []

	def expand(self):
		k = random.choice(self.remaining)
		t = copy.copy(self.tour)
		r = copy.copy(self.remaining)
		t.append(k)
		r.remove(k)
		child = MCTSNode(self, t, r, self.n)
		self.children.append(child)
		return child

	def backprop(self, reward):
		self.visits += 1
		self.total_score += reward
		self.avg_score = self.total_score / self.visits
		if self.parent is not None:
			self.parent.backprop(reward)

	def simulate(self, tsp):
		random.shuffle(self.remaining)
		t = self.tour + self.remaining + [0]
		return tsp.payoff(t)

	def has_children(self):
		return len(self.children) != 0

	def is_leaf(self):
		return len(self.tour) == self.n
	
	def is_fully_expanded(self):
		return len(self.remaining) == len(self.children)

	def get_tour(self):
		return self.tour + [self.tour[0]]

	def best_child_score(self):
		return max(self.children, key = lambda child: child.avg_score)

	def best_child_uct(self):
		k = math.log(self.visits)
		return max(self.children, key = lambda child: child.avg_score + self.c * math.sqrt(2 * k / child.visits))

	def best_child_visits(self):
		return max(self.children, key = lambda child: child.visits)

def mcts(tsp):
	node = MCTSNode(n=tsp.n)
	while not node.is_leaf():
		node = mcts_search(node, tsp)
	mcts_tour = node.get_tour()
	mcts_payoff = tsp.tour_length(mcts_tour)
	return mcts_tour, mcts_payoff

def mcts_search(root_node, tsp, iterations=1000):
	for _ in range(iterations):
		node = tree_policy(root_node)
		pay  = node.simulate(tsp)
		node.backprop(pay)
	return root_node.best_child_score()

def tree_policy(root_node):
	node = root_node
	while not node.is_leaf():
		if not node.is_fully_expanded():
			return node.expand()
		else:
			node = node.best_child_uct()
	return node
