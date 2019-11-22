# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

class LSTM(nn.Module):
	def __init__(self, output_size, hidden_size, input_size):
		"""
		"""
		super(LSTM, self).__init__()

		self.output_size = output_size
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.lstm1 = nn.LSTM(self.input_size, self.hidden_size[0])
		self.batch_norm1 = nn.BatchNorm1d(self.hidden_size[0])
		self.dropout = nn.Dropout(p=0.2)
		self.out_layer = nn.Linear(self.hidden_size[0], self.output_size)
		nn.init.xavier_uniform(self.out_layer.weight)

	def forward(self, input_):
		input_ = input_.permute(1, 0, 2)
		# (num_sequences, batch_size, feature_size)
		output, (final_hidden_state, final_cell_state) = self.lstm1(input_)

		last_hidden = self.batch_norm1(final_hidden_state[0])
		last_hidden = self.dropout(last_hidden)
		# final_hidden_state.size() = (1, batch_size, hidden_size)
		logits = self.out_layer(last_hidden)
		# (batch_size, output_size)

		return logits
