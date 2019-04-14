import numpy as np

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
