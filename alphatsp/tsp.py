import numpy as np

class TSP:

	def __init__(self, n, d=2, points="random_euclidean"):
		self.n, self.d = n, d
		if isinstance(points, (list, np.ndarray)):
			self.points = points
		elif points == "random_euclidean":
			self.points = random_euclidean_tsp(n, d)
		else:
			raise ValueError("Invalid points argument to TSP.")

	def tour_length(self, tour):
		"""Compute the length of the given tour.
		Arguments:
			tour {list(int, n)} -- a permutation of the nodes representing a tour
		Returns:
			float -- tour length
		"""
		points = self.points[tour]
		diffs = np.diff(points, axis=0)
		tour_len = np.linalg.norm(diffs, axis=1, ord=2).sum()
		return tour_len

	def payoff(self, tour):
		"""Compute the payoff of the given tour, a mapping of the tour length to
		[0,1] where 1 is a better tour.
		Arguments:
			tour {list(int, n)} -- a permutation of the nodes representing a tour
		Returns:
			float -- tour length
		"""
		return ((2 * self.n) - self.tour_length(tour)) / (2 * self.n)

def random_euclidean_tsp(n, d=2):
	points = np.random.rand(n, d)
	return points
