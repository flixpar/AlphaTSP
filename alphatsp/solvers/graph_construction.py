import torch
import torch_geometric
from torch_geometric.data import Data

import numpy as np

def construct_graph_grow(tsp, tour, remaining):
	points = torch.tensor(tsp.points).to(dtype=torch.float)

	edges = torch.zeros((2, len(tour)-1), dtype=torch.long)
	for i in range(len(tour)-1):
		edges[0, i] = tour[i]
		edges[1, i] = tour[i+1]

	choices = torch.zeros(tsp.n, dtype=torch.bool)
	choices[remaining] = 1

	x = torch.cat([points, choices.unsqueeze(-1).to(dtype=torch.float)], dim=-1)

	graph = Data(x=x, pos=points, edge_index=edges, y=choices)
	return graph

def construct_graph_prune(tsp, tour, remaining):
	points = torch.tensor(tsp.points).to(dtype=torch.float)
	edges = []

	# construct fully connected on remaining
	for i in range(len(remaining)):
		for j in range(i+1, len(remaining)):
			edges.append((remaining[i], remaining[j]))

	# construct path on tour
	for i in range(len(tour)-1):
		edges.append((tour[i], tour[i+1]))

	edges = torch.tensor(edges, dtype=torch.long).transpose(0,1)

	choices = torch.zeros(tsp.n, dtype=torch.bool)
	choices[remaining] = 1

	x = torch.cat([points, choices.unsqueeze(-1).to(dtype=torch.float)], dim=-1)

	graph = Data(x=x, pos=points, edge_index=edges, y=choices)
	return graph

def construct_graph_prune_weighted(tsp, tour, remaining):
	points = torch.tensor(tsp.points).to(dtype=torch.float)
	edges = []
	edge_lengths = []

	# construct fully connected on remaining
	for i in range(len(remaining)):
		for j in range(i+1, len(remaining)):
			a, b = remaining[i], remaining[j]
			l = np.linalg.norm(tsp.points[a] - tsp.points[b], ord=2)
			edges.append((a, b))
			edge_lengths.append(l)

	# construct path on tour
	for i in range(len(tour)-1):
		a, b = tour[i], tour[i+1]
		l = np.linalg.norm(tsp.points[a] - tsp.points[b], ord=2)
		edges.append((a, b))
		edge_lengths.append(l)

	edges = torch.tensor(edges, dtype=torch.long).transpose(0,1)
	edge_lengths = torch.tensor(edge_lengths, dtype=torch.float).reshape(-1,1)

	choices = torch.zeros(tsp.n, dtype=torch.bool)
	choices[remaining] = 1

	x = torch.cat([points, choices.unsqueeze(-1).to(dtype=torch.float)], dim=-1)

	graph = Data(x=x, pos=points, edge_index=edges, edge_attr=edge_lengths, y=choices)
	return graph