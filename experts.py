import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models

class MLP(nn.module):

	def __init__(self, num_experts=3, max_input_size, hidden_size = 768):
		self.num_experts = num_experts
		self.softmax = nn.Softmax()
		
		self.fc1 = nn.Linear(max_input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, self.num_experts)

	def forward(self, input):

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		out = self.softmax(self.out(x))
		return out


class MoE(nn.Module):
	def __init__(self, num_experts=3, load_gate, input_size=384):
		self.num_experts = num_experts
		if load_gate:
			self.gate = torch.load("save/gate")
		else:
			self.gate = MLP(input_size = input_size)

		self.expert1 = DistilBertForQuestionAnswering.from_pretrained("save/baseline-04/checkpoint")
		self.expert2 = DistilBertForQuestionAnswering.from_pretrained("save/baseline-05/checkpoint") 
		self.expert3 = DistilBertForQuestionAnswering.from_pretrained("save/baseline-06/checkpoint")


