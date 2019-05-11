#include "MCTSNode.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

float tour_len(std::vector<int> tour, std::vector<std::vector<float>> points);

MCTSNode::MCTSNode(int n) {
	this->parent = nullptr;
	this->visits = 0;
	this->total_score = 0.0;
	this->avg_score = 0.0;
	this->n = n;
	this->tour = std::vector<int>();
	this->tour.push_back(0);
	this->remaining = std::set<int>();
	for (int i = 1; i < n; i++)
		this->remaining.insert(i);
}

MCTSNode::MCTSNode(MCTSNode* p, std::vector<int> tour, std::set<int> remaining, int n) {
	this->parent = p;
	this->visits = 0;
	this->total_score = 0.0;
	this->avg_score = 0.0;
	this->n = n;
	this->tour = tour;
	this->remaining = remaining;
}

std::random_device MCTSNode::rd = std::random_device();
std::mt19937 MCTSNode::g = std::mt19937(MCTSNode::rd());

bool MCTSNode::has_children() {
	return this->children.size() > 0;
}

bool MCTSNode::is_leaf() {
	return this->tour.size() == this->n;
}

bool MCTSNode::is_expanded() {
	return this->children.size() == this->remaining.size();
}

std::vector<int> MCTSNode::get_tour() {
	std::vector<int> t(this->tour);
	t.push_back(t[0]);
	return t;
}

std::shared_ptr<MCTSNode> MCTSNode::best_child_score() {
	float best_score = -1;
	std::shared_ptr<MCTSNode> best_node(nullptr);
	for (std::shared_ptr<MCTSNode> n : this->children) {
		if (n->avg_score > best_score) {
			best_score = n->avg_score;
			best_node = n;
		}
	}
	return best_node;
}

std::shared_ptr<MCTSNode> MCTSNode::best_child_visits() {
	float best_score = -1;
	std::shared_ptr<MCTSNode> best_node(nullptr);
	for (std::shared_ptr<MCTSNode> n : this->children) {
		if (n->visits > best_score) {
			best_score = n->visits;
			best_node = n;
		}
	}
	return best_node;
}

std::shared_ptr<MCTSNode> MCTSNode::best_child_uct() {
	float best_score = -1;
	std::shared_ptr<MCTSNode> best_node(nullptr);
	for (std::shared_ptr<MCTSNode> n : this->children) {
		float score = n->avg_score + std::sqrt(2 * std::log(this->visits) / n->visits);
		if (score > best_score) {
			best_score = score;
			best_node = n;
		}
	}
	return best_node;
}

std::shared_ptr<MCTSNode> MCTSNode::expand() {
	std::uniform_int_distribution<> dis(0, this->remaining.size()-1);
	auto it(this->remaining.begin());
	advance(it, dis(g));
	int k = *it;

	std::vector<int> next_tour(this->tour);
	next_tour.push_back(k);

	std::set<int> next_remaining(this->remaining);
	next_remaining.erase(k);

	std::shared_ptr<MCTSNode> m = std::make_shared<MCTSNode>(this, next_tour, next_remaining, this->n);
	this->children.push_back(m);

	return m;
}

void MCTSNode::backprop(float reward) {
	this->visits += 1;
	this->total_score += reward;
	this->avg_score = this->total_score / (float)(this->visits);
	if (this->parent != nullptr) {
		this->parent->backprop(reward);
	}
}

float MCTSNode::simulate(std::vector<std::vector<float>> points) {

	// 1. randomly permute remaining nodes
	std::vector<int> r(this->remaining.begin(), this->remaining.end());
	std::shuffle(r.begin(), r.end(), this->g);

	// 2. merge current tour with permuted remaining nodes
	std::vector<int> sim_tour(this->tour);
	sim_tour.insert(sim_tour.end(), r.begin(), r.end());
	sim_tour.push_back(sim_tour[0]);

	// 3. compute the length of the new tour and return
	float len = tour_len(sim_tour, points);
	return len;

}

float tour_len(std::vector<int> tour, std::vector<std::vector<float>> points) {
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
