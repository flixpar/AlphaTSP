# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.

from math import log, sqrt
import random
import numpy as np
import copy

class TSPState:
	""" A state of the TSP game.
	"""
	def __init__(self, points, tour=[]):
		self.points = points
		self.n = self.points.shape[0]
		self.tour = tour

	def clone(self):
		""" Create a deep clone of this game state.
		"""
		st = TSPState(self.points, copy.copy(self.tour))
		return st

	def do_move(self, move):
		""" Update a state by carrying out the given move.
		"""
		self.tour.append(move)

	def get_moves(self):
		""" Get all possible moves from this state.
		"""
		if len(self.tour) < self.n:
			return list(set(range(self.n)) - set(self.tour))
		elif len(self.tour) == self.n:
			return [self.tour[0]]
		else:
			return []

	def has_moves(self):
		""" Check if there are possible moves in this state.
		"""
		return len(self.tour) <= self.n

	def get_result(self):
		""" Get the game result.
		"""
		return (2 * self.n) - self.get_tour_length()

	def get_tour_length(self):
		""" Get the tour length.
		"""
		points = self.points[self.tour]
		diffs = np.diff(points, axis=0)
		tour_len = np.linalg.norm(diffs, axis=1, ord=2).sum()
		return tour_len

	def __repr__(self):
		s = f"Tour: {self.tour}"
		return s


class Node:
	""" A node in the game tree.
	"""
	def __init__(self, move = None, parent = None, state = None):
		self.move = move # the move that got us to this node - "None" for the root node
		self.parentNode = parent # "None" for the root node
		self.childNodes = []
		self.total_score = 0
		self.avg_score = 0
		self.visits = 0
		self.untriedMoves = state.get_moves() # future child nodes

	def uct_select_child(self):
		""" Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
			lambda c: c.total_score/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
			exploration versus exploitation.
		"""
		s = sorted(self.childNodes, key = lambda c: c.avg_score + sqrt(2*log(self.visits)/c.visits))[-1]
		return s

	def add_child(self, m, s):
		""" Remove m from untriedMoves and add a new child node for this move.
			Return the added child node
		"""
		n = Node(move = m, parent = self, state = s)
		self.untriedMoves.remove(m)
		self.childNodes.append(n)
		return n

	def update(self, result):
		""" Update this node - one additional visit and result additional score.
		"""
		self.visits += 1
		self.total_score += result
		self.avg_score = self.total_score / self.visits

	def __repr__(self):
		return "[M:" + str(self.move) + " S/V:" + str(self.avg_score) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

	def tree_to_string(self, indent):
		s = self.indent_string(indent) + str(self)
		for c in self.childNodes:
			s += c.tree_to_string(indent+1)
		return s

	def indent_string(self,indent):
		s = "\n"
		for i in range (1,indent+1):
			s += "| "
		return s

	def children_to_string(self):
		s = ""
		for c in self.childNodes:
			s += str(c) + "\n"
		return s


def UCT(rootstate, itermax, verbose = False):
	""" Conduct a UCT search for itermax iterations starting from rootstate.
		Return the best move from the rootstate."""

	rootnode = Node(state = rootstate)

	for i in range(itermax):
		node = rootnode
		state = rootstate.clone()

		# Select
		while (not node.untriedMoves) and node.childNodes: # node is fully expanded and non-terminal
			node = node.uct_select_child()
			state.do_move(node.move)

		# Expand
		if node.untriedMoves: # if we can expand (i.e. state/node is non-terminal)
			m = random.choice(node.untriedMoves)
			state.do_move(m)
			node = node.add_child(m,state) # add child and descend tree

		# Rollout
		while state.has_moves(): # while state is non-terminal
			state.do_move(random.choice(state.get_moves()))

		# Backpropagate
		while node != None: # backpropagate from the expanded node and work back to the root node
			node.update(state.get_result()) # state is terminal. Update node with result.
			node = node.parentNode

	# Output some information about the tree
	if (verbose):
		print(rootnode.tree_to_string(0))
		print(rootnode.children_to_string())

	return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited
