import os
import datetime
import csv
import shutil
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class Logger:

	def __init__(self, enabled=True):
		self.logging = enabled
		if not self.logging:
			return

		self.dt = datetime.datetime.now().strftime("%m%d_%H%M")
		self.path = f"./saves/{self.dt}"

		if not os.path.exists(self.path):
			os.makedirs(self.path)

		self.losses = []
		self.eval = []

		self.main_log_fn = os.path.join(self.path, "log.txt")
		shutil.copy2("args.py", self.path)

	def save_model(self, model, iterations):
		if not self.logging: return
		if isinstance(iterations, int):
			fn = os.path.join(self.path, f"policynet_{iterations:07d}.pth")
		else:
			fn = os.path.join(self.path, f"policynet_{iterations}.pth")
		torch.save(model.state_dict(), fn)
		self.print(f"Saved model to: {fn}\n")

	def print(self, *x):
		print(*x)
		self.log(*x)

	def log(self, *x):
		if not self.logging: return
		with open(self.main_log_fn, "a") as f:
			print(*x, file=f, flush=True)

	def log_loss(self, l):
		self.losses.append(l)

	def log_eval(self, data):
		if not self.logging: return
		self.eval.append(data)

	def save(self):
		if not self.logging: return

		with open(os.path.join(self.path, "loss.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it", "loss"])
			csvwriter.writeheader()
			for it, loss in enumerate(self.losses):
				row = {"it": it, "loss": loss}
				csvwriter.writerow(row)

		if self.eval:
			with open(os.path.join(self.path, "eval.csv"), "w") as f:
				cols = ["it"] + sorted(list(set(self.eval[0].keys()) - set(["it"])))
				csvwriter = csv.DictWriter(f, cols)
				csvwriter.writeheader()
				for row in self.eval_scores:
					csvwriter.writerow(row)

		plt.clf()

		plt.plot(self.losses)
		plt.xlabel("iterations")
		plt.ylabel("training loss")
		plt.savefig(os.path.join(self.path, "losses.png"))
