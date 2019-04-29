# Copyright 2019, Gurobi Optimization, LLC

import math
import random
import itertools
import gurobipy as gbp

from alphatsp.util import stdout_redirected, stderr_redirected

n = 0

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
	if where == gbp.GRB.Callback.MIPSOL:
		# make a list of edges selected in the solution
		vals = model.cbGetSolution(model._vars)
		selected = gbp.tuplelist((i,j) for i,j in model._vars.keys() if vals[i,j] > 0.5)
		# find the shortest cycle in the selected edge list
		tour = subtour(selected)
		if len(tour) < n:
			# add subtour elimination constraint for every pair of cities in tour
			model.cbLazy(gbp.quicksum(model._vars[i,j] for i,j in itertools.combinations(tour, 2)) <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
	unvisited = list(range(n))
	cycle = range(n+1) # initial length has 1 more city
	while unvisited: # true if list is non-empty
		thiscycle = []
		neighbors = unvisited
		while neighbors:
			current = neighbors[0]
			thiscycle.append(current)
			unvisited.remove(current)
			neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]
		if len(cycle) > len(thiscycle):
			cycle = thiscycle
	return cycle

def exact_gurobi(tsp):

	global n
	points, n = tsp.points, tsp.n

	# Dictionary of Euclidean distance between each pair of points
	dist = {(i,j) : math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2))) for i in range(n) for j in range(i)}

	with stdout_redirected(), stderr_redirected():

		m = gbp.Model()

		# Create variables
		vars = m.addVars(dist.keys(), obj=dist, vtype=gbp.GRB.BINARY, name='e')
		for i,j in vars.keys():
			vars[j,i] = vars[i,j] # edge in opposite direction

		# Add degree-2 constraint
		m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

		# Optimize model
		m._vars = vars
		m.Params.lazyConstraints = 1
		m.optimize(subtourelim)

		vals = m.getAttr('x', vars)
		selected = gbp.tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

	tour = subtour(selected)
	assert len(tour) == n

	tour.append(tour[0])
	tour_len = tsp.tour_length(tour)

	return tour, tour_len
