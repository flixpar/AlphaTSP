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
        s = -1
        
        for i in nontour_nodes:
            
            dists = distances[i, tour_nodes]
            opt_ind = np.argmin(dists)
            opt_dist = dists[opt_ind]
            
            if opt_dist < best_dist:
                best_dist = opt_dist
                s = i
                
        tour_nodes.append(s)
        nontour_nodes.remove(s)
        
        best_dist = np.inf
        e = (-1, -1)
        
        for j in tour_nodes:
        
            v_prev = np.argmax(adj[:,j])
            v_next = np.argmax(adj[j,:])

            inc1 = distances[v_prev, s] + distances[s, j] - distances[v_prev, j]
            inc2 = distances[j, s] + distances[s, v_next] - distances[j, v_next]
            
            if inc1 < best_dist or inc2 < best_dist:
                if inc1 <= inc2:
                    e = (v_prev, j)
                    best_dist = inc1
                else:
                    e = (j, v_next)
                    best_dist = inc2
                
        v_prev, v_next = e
        adj[v_prev, s] = 1
        adj[s, v_next] = 1
        adj[v_prev, v_next] = 0

    tour = [0]
    while len(tour) <= tsp.n:
        tour.append(np.argmax(adj[tour[-1], :]))

    return tour

def farthest_insertion(tsp):
    
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
        
        best_dist = 0
        s = -1
        
        for i in nontour_nodes:
            
            dists = distances[i, tour_nodes]
            opt_ind = np.argmax(dists)
            opt_dist = dists[opt_ind]
            
            if opt_dist > best_dist:
                best_dist = opt_dist
                s = i
                
        tour_nodes.append(s)
        nontour_nodes.remove(s)
        
        best_dist = np.inf
        e = (-1, -1)
        
        for j in tour_nodes:
        
            v_prev = np.argmax(adj[:,j])
            v_next = np.argmax(adj[j,:])

            inc1 = distances[v_prev, s] + distances[s, j] - distances[v_prev, j]
            inc2 = distances[j, s] + distances[s, v_next] - distances[j, v_next]
            
            if inc1 < best_dist or inc2 < best_dist:
                if inc1 <= inc2:
                    e = (v_prev, j)
                    best_dist = inc1
                else:
                    e = (j, v_next)
                    best_dist = inc2
                
        v_prev, v_next = e
        adj[v_prev, s] = 1
        adj[s, v_next] = 1
        adj[v_prev, v_next] = 0

    tour = [0]
    while len(tour) <= tsp.n:
        tour.append(np.argmax(adj[tour[-1], :]))

    return tour
