from concorde.tsp import TSPSolver
from alphatsp.util import stderr_redirected, stdout_redirected

def exact(tsp):

	points = tsp.points * 1000
	n, d = points.shape

	if d != 2:
		raise Exception(f"Concorde solver currently only supports 2D points. {d}D points given.")
	
	xs = points[:, 0]
	ys = points[:, 1]
	norm_type = "EUC_2D"

	with stdout_redirected(), stderr_redirected():

		concorde_tsp_instance = TSPSolver.from_data(xs, ys, norm_type)
		tour, val, success, foundtour, hit_timebound = concorde_tsp_instance.solve()

	if not success:   print("WARNING: Concorde solver failed.")
	if not foundtour: print("WARNING: Concorde solver did not find a tour.")
	if hit_timebound: print("WARNING: Concorde solver ran out of time.")

	tour = tour.tolist()
	tour.append(tour[0])
	tour_len = tsp.tour_length(tour)

	return tour, tour_len
