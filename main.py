import argparse
from alphatsp.experiments import *

def main(args):
	if args.experiment == "nearest_greedy":
		nearest_greedy.run()
	elif args.experiment == "mcts":
		mcts_test.run()
	else:
		raise ValueError("Invalid experiment selection.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, required=True, help="experiment name")
	args = parser.parse_args()
	main(args)
