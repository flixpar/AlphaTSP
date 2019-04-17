all: mcts

mcts: mcts.o mctsnode.o
	g++ --std=c++17 -O3 -o mcts mcts.o mctsnode.o

mcts.o: mcts.cpp
	g++ --std=c++17 -O3 -c -o mcts.o mcts.cpp

mctsnode.o: MCTSNode.cpp MCTSNode.h
	g++ --std=c++17 -O3 -c -o mctsnode.o MCTSNode.cpp

clean:
	rm *.o mcts