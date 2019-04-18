import matplotlib
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

import os, sys
from contextlib import contextmanager

def display_tour(tsp, tour):
	points = tsp.points
	points = points[tour]
	plt.plot(points[:,0], points[:,1], 'o-')
	plt.title("TSP Tour - Nearest Neighbor Greedy")
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()

@contextmanager
def stdout_redirected(to=os.devnull):
	fd = sys.stdout.fileno()
	def _redirect_stdout(to):
		sys.stdout.close()
		os.dup2(to.fileno(), fd)
		sys.stdout = os.fdopen(fd, 'w')
	with os.fdopen(os.dup(fd), 'w') as old_stdout:
		with open(to, 'w') as file:
			_redirect_stdout(to=file)
		try:
			yield
		finally:
			_redirect_stdout(to=old_stdout)
