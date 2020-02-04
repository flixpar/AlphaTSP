import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, ARMAConv, XConv, SAGEConv
from torch_geometric.data import Data, DataLoader

from alphatsp.logger import Logger

if torch.cuda.is_available(): device = torch.device("cuda:0")
else:                         device = torch.device("cpu")


class GCNPolicyNetwork(nn.Module):
	def __init__(self, d=3):
		super(GCNPolicyNetwork, self).__init__()
		self.conv1 = GCNConv(d,  16, improved=True, cached=True)
		self.conv2 = GCNConv(16, 16, improved=True, cached=True)
		self.conv3 = GCNConv(16,  1, improved=True, cached=True)
		self.fc    = nn.Linear(16, 1)

	def forward(self, graph):
		x, edges, choices = graph.x, graph.edge_index, graph.y

		x = self.conv1(x, edges)
		x = F.relu(x)
		x = self.conv2(x, edges)
		x = F.relu(x)

		c = self.conv3(x, edges)
		choice = torch.masked_select(c.squeeze(), choices)
		choice = F.softmax(choice, dim=0)

		v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device))
		value = self.fc(v)

		return choice, value

class ARMAPolicyNetwork(torch.nn.Module):
	def __init__(self, d=3):
		super(ARMAPolicyNetwork, self).__init__()

		self.conv1 = ARMAConv(
			d,
			16,
			num_stacks=3,
			num_layers=2,
			shared_weights=True,
			dropout=0.1)

		self.conv2 = ARMAConv(
			16,
			16,
			num_stacks=3,
			num_layers=2,
			shared_weights=True,
			dropout=0.1,
			act=None)

		self.conv3 = ARMAConv(
			16,
			1,
			num_stacks=3,
			num_layers=2,
			shared_weights=True,
			dropout=0.1,
			act=None).to(device)

		self.fc = nn.Linear(16, 1).to(device)

	def forward(self, graph):
		x, edges, choices = graph['x'], graph['edge_index'], graph['y']

		x = self.conv1(x, edges)
		x = F.relu(x)
		x = self.conv2(x, edges)
		x = F.relu(x)

		c = self.conv3(x, edges)
		choice = torch.masked_select(c.squeeze(), choices)
		choice = F.softmax(choice, dim=0)

		v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device))
		value = self.fc(v)

		return choice, value

class SagePolicyNetwork(nn.Module):
	def __init__(self, d=3):
		super(SagePolicyNetwork, self).__init__()
		self.conv1 = SAGEConv(d,  16)
		self.conv2 = SAGEConv(16, 16)
		self.conv3 = SAGEConv(16, 1)
		self.fc    = nn.Linear(16, 1)

	def forward(self, graph):
		x, edges, choices = graph.x, graph.edge_index, graph.y

		x = self.conv1(x, edges)
		x = F.relu(x)
		x = self.conv2(x, edges)
		x = F.relu(x)

		c = self.conv3(x, edges)
		choice = torch.masked_select(c.squeeze(), choices)
		choice = F.softmax(choice, dim=0)

		v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device))
		value = self.fc(v)

		return choice, value

class WeightedGCNPolicyNetwork(nn.Module):
	def __init__(self, d=3):
		super(WeightedGCNPolicyNetwork, self).__init__()
		self.conv1 = GCNConv(d,  16, improved=True, cached=True)
		self.conv2 = GCNConv(16, 16, improved=True, cached=True)
		self.conv3 = GCNConv(16,  1, improved=True, cached=True)
		self.fc    = nn.Linear(16, 1)

	def forward(self, graph):
		x, edges, edge_feat, choices = graph.x, graph.edge_index, graph.edge_attr, graph.y

		x = self.conv1(x, edges, edge_feat)
		x = F.relu(x)
		x = self.conv2(x, edges, edge_feat)
		x = F.relu(x)

		c = self.conv3(x, edges, edge_feat)
		choice = torch.masked_select(c.squeeze(), choices)
		choice = F.softmax(choice, dim=0)

		v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device))
		value = self.fc(v)

		return choice, value

class PointCNNPolicyNetwork(nn.Module):
	def __init__(self, d=3):
		super(PointCNNPolicyNetwork, self).__init__()
		self.conv1 = XConv(d,  16, dim=2, kernel_size=10, hidden_channels=4)
		self.conv2 = XConv(16, 16, dim=2, kernel_size=10, hidden_channels=4)
		self.conv3 = XConv(16,  1, dim=2, kernel_size=10, hidden_channels=4)
		self.fc    = nn.Linear(16, 1)

	def forward(self, graph):
		x, pos, choices = graph.x, graph.pos, graph.y

		x = self.conv1(x, pos)
		x = F.relu(x)
		x = self.conv2(x, pos)
		x = F.relu(x)

		c = self.conv3(x, pos)
		choice = torch.masked_select(c.squeeze(), choices)
		choice = F.softmax(choice, dim=0)

		v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device))
		value = self.fc(v)

		return choice, value

class PolicyNetworkTrainer:

	def __init__(self, model, example_queue):

		self.model = model
		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-5)

		self.example_queue = example_queue
		self.losses = []
		self.n_examples_used = 0

	def train_example(self):
		self.model.train()

		example = self.example_queue.get()
		if example is None: return -1
		graph, choice_probs, value = example["graph"], example["choice_probs"], example["pred_value"]

		graph = Data(**graph)
		graph = graph.to(device)

		pred_choices, pred_value = self.model(graph)
		loss = self.loss_fn(pred_choices, choice_probs) + (0.2 * self.loss_fn(pred_value, value))

		self.losses.append(loss.item())

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.n_examples_used += 1
		return 0

	def train_all(self):
		self.model.train()
		while not self.example_queue.empty():
			ret_code = self.train_example()
			if ret_code == -1: return -1
		return 0

	def save_model(self):
		torch.save(self.model.state_dict(), f"saves/policynet_{self.n_examples_used:06d}.pth")

class SupervisedPolicyNetworkTrainer:

	def __init__(self, model, example_queue):

		self.model = model.to(device)
		self.value_loss_fn = nn.MSELoss()
		self.choice_loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-5)

		self.example_queue = example_queue
		self.n_examples_used = 0

		self.logger = Logger()

	def train_all(self):
		while True:
			if not self.example_queue.empty():
				return_code = self.train_example()
				if self.n_examples_used%1000 == 0 and self.n_examples_used!=0:
					self.logger.print(f"iter={self.n_examples_used}, avg_loss={sum(self.logger.losses[-100:])/100:.4f}")
				if self.n_examples_used%10000 == 0 and self.n_examples_used!=0:
					self.logger.save_model(self.model, self.n_examples_used)
				if return_code == -1:
					self.logger.save()
					self.logger.save_model(self.model, "final")
					return

	def train_example(self):
		self.model.train()

		example = self.example_queue.get()
		if example is None: return -1
		graph, choice, value = example["graph"], example["choice"], example["pred_value"]

		graph = Data(**graph)
		graph = graph.to(device)

		pred_choices, pred_value = self.model(graph)
		choice, value = torch.tensor([choice], device=device), torch.tensor([value], device=device)
		pred_choices, pred_value = pred_choices.unsqueeze(0).to(device), pred_value.squeeze(0).to(device)
		loss = self.choice_loss_fn(pred_choices, choice) + 0.2 * self.value_loss_fn(pred_value, value)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.logger.log_loss(loss.item())
		self.n_examples_used += 1
		return 0
