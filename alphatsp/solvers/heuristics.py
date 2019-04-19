import numpy as np
import scipy.spatial.distance

def nearest_greedy(tsp):
	start = np.random.randint(tsp.n)
	tour = [start]
	remaining = list(set(range(tsp.n)) - set(tour))
	while remaining:
		next_remaining = np.argmin(np.linalg.norm(tsp.points[tour[-1]] - tsp.points[remaining], ord=2, axis=1))
		next_node = remaining[next_remaining]
		tour.append(next_node)
		remaining = list(set(range(tsp.n)) - set(tour))
	tour.append(start)
	return tour, tsp.tour_length(tour)

def nearest_insertion(tsp):

	distances = scipy.spatial.distance.pdist(tsp.points, metric="euclidean")
	distances = scipy.spatial.distance.squareform(distances)
	np.fill_diagonal(distances, np.inf)

	adj = np.zeros((tsp.n, tsp.n), dtype=np.bool)
	start_node = np.argmax(distances[0, 1:]) + 1
	adj[0, start_node] = 1
	adj[start_node, 0] = 1

	nontour_nodes = set(range(tsp.n))
	tour_nodes = list()

	nontour_nodes.remove(0)
	nontour_nodes.remove(start_node)
	tour_nodes.append(0)
	tour_nodes.append(start_node)

	while nontour_nodes:

		best_dist = np.inf
		best_edge = (-1, -1)

		for i in nontour_nodes:

			dists = distances[i, tour_nodes]
			opt_ind = np.argmin(dists)
			opt_dist = dists[opt_ind]
			opt_node = tour_nodes[opt_ind]

			if opt_dist < best_dist:
				best_dist = opt_dist
				best_edge = (i, opt_node)

		i, j = best_edge

		tour_nodes.append(i)
		nontour_nodes.remove(i)

		egde_in = np.argmax(adj[:,j])
		edge_out = np.argmax(adj[j,:])

		inc1 = distances[egde_in, i] + distances[i, j] - distances[egde_in, j]
		inc2 = distances[j, i] + distances[i, edge_out] - distances[j, edge_out]

		if inc1 <= inc2:
			adj[egde_in, j] = 0
			adj[egde_in, i] = 1
			adj[i, j] = 1
		else:
			adj[j, edge_out] = 0
			adj[i, edge_out] = 1
			adj[j, i] = 1

	tour = [0]
	while len(tour) <= tsp.n:
		tour.append(np.argmax(adj[tour[-1], :]))

	return tour, tsp.tour_length(tour)

def farthest_insertion(tsp):

	distances = scipy.spatial.distance.pdist(tsp.points, metric="euclidean")
	distances = scipy.spatial.distance.squareform(distances)
	np.fill_diagonal(distances, 0)

	adj = np.zeros((tsp.n, tsp.n), dtype=np.bool)
	start_node = np.argmax(distances[0, 1:]) + 1
	adj[0, start_node] = 1
	adj[start_node, 0] = 1

	nontour_nodes = set(range(tsp.n))
	tour_nodes = list()

	nontour_nodes.remove(0)
	nontour_nodes.remove(start_node)
	tour_nodes.append(0)
	tour_nodes.append(start_node)

	while nontour_nodes:

		best_dist = 0
		best_edge = (-1, -1)

		for i in nontour_nodes:

			dists = distances[i, tour_nodes]
			opt_ind = np.argmax(dists)
			opt_dist = dists[opt_ind]
			opt_node = tour_nodes[opt_ind]

			if opt_dist > best_dist:
				best_dist = opt_dist
				best_edge = (i, opt_node)

		i, j = best_edge

		tour_nodes.append(i)
		nontour_nodes.remove(i)

		egde_in = np.argmax(adj[:,j])
		edge_out = np.argmax(adj[j,:])

		inc1 = distances[egde_in, i] + distances[i, j] - distances[egde_in, j]
		inc2 = distances[j, i] + distances[i, edge_out] - distances[j, edge_out]

		if inc1 <= inc2:
			adj[egde_in, j] = 0
			adj[egde_in, i] = 1
			adj[i, j] = 1
		else:
			adj[j, edge_out] = 0
			adj[i, edge_out] = 1
			adj[j, i] = 1

	tour = [0]
	while len(tour) <= tsp.n:
		tour.append(np.argmax(adj[tour[-1], :]))

	return tour, tsp.tour_length(tour)
