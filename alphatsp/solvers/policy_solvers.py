from alphatsp.solvers.mcts import MCTSNode, MCTSTree, MCTSSolver

class PolicySolver:

	def __init__(self, args, tsp, model):
		self.tsp = tsp
		self.tree = MCTSTree(args, self.tsp)
		self.root_node = self.tree.root_node
		self.model = model

	def solve(self):
		node = self.root_node
		while not node.is_leaf():
			node = node.best_remaining_policy(model=self.model)
		mcts_tour = node.get_tour()
		mcts_payoff = self.tsp.tour_length(mcts_tour)
		return mcts_tour, mcts_payoff

class PolicyMCTSSolver(MCTSSolver):

	def __init__(self, args, tsp, model):
		self.model = model
		selection_func = lambda node: node.select_child_policy(self.model)
		super(PolicyMCTSSolver, self).__init__(args, tsp, selection_func=selection_func)
