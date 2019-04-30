import argparse
from alphatsp.experiments import (
	nearestneighbor,
	mcts,
	exact,
	gurobi,
	insertion,
	policy,
	parallel
)

def main(args):
	if args.experiment == "greedy":
		nearestneighbor.run()
	elif args.experiment == "mcts":
		mcts.run()
	elif args.experiment == "exact":
		exact.run()
	elif args.experiment == "gurobi":
		gurobi.run()
	elif args.experiment == "insertion":
		insertion.run()
	elif args.experiment == "policy":
		policy.run()
	elif args.experiment == "parallel":
		parallel.run()
	else:
		raise ValueError("Invalid experiment selection.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, required=True, help="experiment name")
	args = parser.parse_args()
	main(args)
