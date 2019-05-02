import argparse
import multiprocessing as mp
from args import Args
from alphatsp.experiments import (
	nearestneighbor,
	mcts,
	exact,
	gurobi,
	insertion,
	policy,
	parallel,
	selfplay
)

def main(args):
	a = Args()
	if args.experiment == "greedy":
		nearestneighbor.run(a)
	elif args.experiment == "mcts":
		mcts.run(a)
	elif args.experiment == "exact":
		exact.run(a)
	elif args.experiment == "gurobi":
		gurobi.run(a)
	elif args.experiment == "insertion":
		insertion.run(a)
	elif args.experiment == "policy":
		policy.run(a)
	elif args.experiment == "parallel":
		parallel.run(a)
	elif args.experiment == "selfplay":
		selfplay.run(a)
	else:
		raise ValueError("Invalid experiment selection.")

if __name__ == "__main__":
	mp.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, required=True, help="experiment name")
	args = parser.parse_args()
	main(args)
