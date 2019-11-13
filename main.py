import argparse
from args import Args
import importlib

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def main(args):
	config = Args()
	experiment = importlib.import_module(f"alphatsp.experiments.{args.experiment}")
	experiment.run(config)

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, required=True, help="experiment name")
	args = parser.parse_args()
	main(args)
