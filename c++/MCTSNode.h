#ifndef MCTSNODEH
#define MCTSNODEH

#include <random>
#include <vector>
#include <set>

class MCTSNode {
private:
	MCTSNode* parent;
	std::vector<int> tour;
	std::set<int> remaining;
	int visits;
	float total_score;
	float avg_score;
	int n;
	static std::random_device rd;
	static std::mt19937 g;
public:
	MCTSNode(int n);
	MCTSNode(MCTSNode* p, std::vector<int> tour, std::set<int> remaining, int n);
	std::shared_ptr<MCTSNode> expand();
	void backprop(float reward);
	float simulate(std::vector<std::vector<float>> points);
	std::vector<int> get_tour();
	bool has_children();
	bool is_leaf();
	bool is_expanded();
	std::shared_ptr<MCTSNode> best_child_score();
	std::shared_ptr<MCTSNode> best_child_visits();
	std::shared_ptr<MCTSNode> best_child_uct();
	std::vector<std::shared_ptr<MCTSNode>> children;
};

#endif