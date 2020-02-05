class Args:

	N = 30
	D = 2

	mcts_iters = 3000
	exploration_constant = 1.0

	n_train_examples = 50_000
	n_test_examples  = 10

	policy_network = "arma"
	graph_construction = "prune_weighted"

	n_threads = 8
