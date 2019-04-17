import numpy as np
import copy
import random

class MCTSNode:

	def __init__(self, p=None, t=[], r=None, n=-1):
		self.parent = p
		self.tour = t
		self.remaining = r if r is not None else set(range(n))
		self.visits = 0
		self.total_score = 0
		self.avg_score = 0
		self.n = n
		self.children = []

	def expand(self):
		for k in self.remaining:
			t = copy.copy(self.tour)
			r = copy.copy(self.remaining)
			t.append(k)
			r.remove(k)
			self.children.append(MCTSNode(self, t, r, self.n))

	def backprop(self, reward):
		self.visits += 1
		self.total_score += reward
		self.avg_score = self.total_score / self.visits
		if self.parent is not None:
			self.parent.backprop(reward)

	def simulate(self, tsp):
		r = list(self.remaining)
		random.shuffle(r)
		t = self.tour + r + [self.tour[0]]
		return tsp.payoff(t)

	def has_children(self):
		return len(self.children) != 0

	def is_leaf(self):
		return len(self.tour) == self.n

	def get_tour(self):
		return self.tour + [self.tour[0]]

	def best_child_score(self):
		scores = [child.avg_score for child in self.children]
		scores = np.asarray(scores)
		return self.children[np.argmax(scores)]

	def best_child_uct(self):
		k = np.log(self.visits)
		scores = [child.avg_score + np.sqrt(2 * k / child.visits) for child in self.children]
		scores = np.asarray(scores)
		return self.children[np.argmax(scores)]

	def best_child_visits(self):
		visits = [child.visits for child in self.children]
		visits = np.asarray(visits)
		return self.children[np.argmax(visits)]


def mcts(root, tsp, iterations=1000):
	node = root
	for _ in range(iterations):
		while node.has_children():
			node = node.best_child_uct()
		node.expand()
		for child in node.children:
			pay = child.simulate(tsp)
			child.backprop(pay)
	return root.best_child_score()
