#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include "MCTSNode.h"

std::shared_ptr<MCTSNode> mcts(std::shared_ptr<MCTSNode> rootnode, std::vector<std::vector<float>> points, int iterations);
float compute_tour_length(std::vector<int> tour, std::vector<std::vector<float>> points);
std::vector<int> greedy(std::vector<std::vector<float>> points);
void random_tours(std::vector<std::vector<float>> points);

int main() {

	// 1. Create TSP instance

	int n = 60;
	int d = 2;
	int iterations = 1000;

	std::vector<std::vector<float>> points;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	for (int i = 0; i < n; i++) {
		std::vector<float> p = {(float)dis(gen), (float)dis(gen)};
		points.push_back(p);
	}

	// 2. Construct MCTS tree

	std::shared_ptr<MCTSNode> rootnode = std::make_shared<MCTSNode>(n);
	std::shared_ptr<MCTSNode> node(rootnode);

	// 3. Run MCTS at each level of the tree

	while (!node->is_leaf()) {
		node = mcts(node, points, iterations);
	}

	// 4. Display result

	std::vector<int> optimal_tour(node->get_tour());
	float optimal_tour_length = compute_tour_length(optimal_tour, points);

	for (int i = 0; i < optimal_tour.size(); i++) {
		std::cout << optimal_tour[i];
		if (i != optimal_tour.size() - 1) {
			std::cout << " -> ";
		}
	}
	std::cout << std::endl;
	std::cout << "Tour length: " << optimal_tour_length << std::endl;

	// 5. Run greedy

	std::vector<int> greedy_tour = greedy(points);
	float greedy_tour_length = compute_tour_length(greedy_tour, points);

	for (int i = 0; i < greedy_tour.size(); i++) {
		std::cout << greedy_tour[i];
		if (i != greedy_tour.size() - 1) {
			std::cout << " -> ";
		}
	}
	std::cout << std::endl;
	std::cout << "Greedy tour length: " << greedy_tour_length << std::endl;

	// 6. Random tours
	random_tours(points);

	// 7. Return
	return 0;
}

std::shared_ptr<MCTSNode> mcts(std::shared_ptr<MCTSNode> rootnode, std::vector<std::vector<float>> points, int iterations) {
	
	int n = points.size();

	// 1. Begin search
	for (int it=0; it < iterations; it++) {

		std::shared_ptr<MCTSNode> node(rootnode);

		// 2. Descend
		while (!node->is_leaf()) {
			if (!node->is_expanded()) {
				node = node->expand();
				break;
			} else {
				node = node->best_child_uct();
			}
		}

		// 3. Simulate
		float tour_len = node->simulate(points);
		float reward = ((2.0 * n) - tour_len) / (2.0 * n);

		// 4. Backprop
		node->backprop(reward);

	}

	// 5. Select and return best child node
	return rootnode->best_child_score();

}

float compute_tour_length(std::vector<int> tour, std::vector<std::vector<float>> points) {
	float len = 0;
	int d = points[0].size();
	int n = points.size();
	for (int i = 1; i < n+1; i++) {
		float edge_len = 0;
		for (int j = 0; j < d; j++) {
			float diff = points[tour[i]][j] - points[tour[i-1]][j];
			diff = diff * diff;
			edge_len += diff;
		}
		edge_len = std::sqrt(edge_len);
		len += edge_len;
	}
	return len;
}

std::vector<int> greedy(std::vector<std::vector<float>> points) {

	// 1. Get start node
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, points.size()-1);
	int start = dis(gen);

	// 2. Start tour
	std::vector<int> tour = {start};

	// 3. Compute remaining
	std::set<int> remaining;
	for (int i = 0; i < points.size(); i++) {
		if (i != start) {
			remaining.insert(i);
		}
	}

	// 4. Build tour
	std::vector<float> pt1 = points[start];
	while (!remaining.empty()) {

		// 4.1 Compute min distance
		int next_node = -1;
		float min_dist = std::numeric_limits<float>::max();
		for (int ind2 : remaining) {

			std::vector<float> pt2 = points[ind2];

			float edge_len = 0;
			for (int j = 0; j < pt1.size(); j++) {
				float diff = pt1[j] - pt2[j];
				diff = diff * diff;
				edge_len += diff;
			}
			edge_len = std::sqrt(edge_len);

			if (edge_len < min_dist) {
				min_dist = edge_len;
				next_node = ind2;
			}

		}

		// 4.2 Add to tour, remove from remaining
		pt1 = points[next_node];
		tour.push_back(next_node);
		remaining.erase(next_node);

	}

	// 5. Complete tour
	tour.push_back(tour[0]);

	// 6. Return tour
	return tour;

}

void random_tours(std::vector<std::vector<float>> points) {

	int iterations = 100000;

	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<int> tour;
	for (int i = 0; i < points.size(); i++) {
		tour.push_back(i);
	}

	float total_len = 0;
	float best_len = std::numeric_limits<float>::max();

	for (int i = 0; i < iterations; i++) {
		std::vector<int> t(tour);
		std::shuffle(t.begin(), t.end(), gen);
		t.push_back(t[0]);
		float l = compute_tour_length(t, points);
		total_len += l;
		if (l < best_len)
			best_len = l;
	}

	float avg_len = total_len / (float)iterations;

	std::cout << "Random avg length: " << avg_len << std::endl;
	std::cout << "Random best length: " << best_len << std::endl;

}
