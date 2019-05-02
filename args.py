class Args:

	N = 30
	D = 2

	mcts_iters = 3000
	exploration_constant = 0.7

	n_train_examples = 10_000
	n_test_examples  = 50

	policy_network = "gcn"
	graph_construction = "grow"
	weighted_graph = False

	n_threads = 8
