import matplotlib
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

def display_tour(tsp, tour):
	points = tsp.points
	points = points[tour]
	plt.plot(points[:,0], points[:,1], 'o-')
	plt.title("TSP Tour - Nearest Neighbor Greedy")
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
