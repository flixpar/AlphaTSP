import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

import alphatsp.solvers.policy_networks
import alphatsp.solvers.graph_construction

import os, sys
from contextlib import contextmanager

def get_policy_network(network_type):
	if network_type == "gcn":
		return alphatsp.solvers.policy_networks.GCNPolicyNetwork()
	elif network_type == "arma":
		return alphatsp.solvers.policy_networks.ARMAPolicyNetwork()
	elif network_type == "pointcnn":
		return alphatsp.solvers.policy_networks.PointCNNPolicyNetwork()
	elif network_type == "gcn_weighted":
		return alphatsp.solvers.policy_networks.WeightedGCNPolicyNetwork()
	else:
		raise ValueError("Invalid network type given.")

def get_graph_constructor(construction_type):
	if construction_type == "grow":
		return alphatsp.solvers.graph_construction.construct_graph_grow
	elif construction_type == "prune":
		return alphatsp.solvers.graph_construction.construct_graph_prune
	elif construction_type == "prune_weighted":
		return alphatsp.solvers.graph_construction.construct_graph_prune_weighted
	else:
		raise ValueError("Invalid graph construction type given.")

def display_tour(tsp, tour, title=""):
	points = tsp.points
	points = points[tour]
	plt.plot(points[:,0], points[:,1], 'o-')
	plt.title(title)
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

@contextmanager
def stderr_redirected(to=os.devnull):
	fd = sys.stderr.fileno()
	def _redirect_stderr(to):
		sys.stderr.close()
		os.dup2(to.fileno(), fd)
		sys.stderr = os.fdopen(fd, 'w')
	with os.fdopen(os.dup(fd), 'w') as old_stderr:
		with open(to, 'w') as file:
			_redirect_stderr(to=file)
		try:
			yield
		finally:
			_redirect_stderr(to=old_stderr)
