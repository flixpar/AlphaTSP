import numpy as np

class TSP:

	def __init__(self, n, d=2, points="random_euclidean"):
		self.n, self.d = n, d
		self.points = random_euclidean_tsp(n, d)

	def tour_length(self, tour):
		"""Compute the length of the given tour.
		Arguments:
			tour {list(int, n)} -- a permutation of the nodes representing a tour
		Returns:
			int -- tour length
		"""
		tour_len = 0
		for i in range(1, len(tour)):
			tour_len += np.linalg.norm(self.points[tour[i]] - self.points[tour[i-1]], ord=2)
		return tour_len

def random_euclidean_tsp(n, d=2):
	points = np.random.rand(n, d)
	return points
