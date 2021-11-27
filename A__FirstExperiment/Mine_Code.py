from __future__ import unicode_literals, print_function, division

import unicodedata
import string
import re
import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from io import open
from torch import optim
from config import Config

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config['BatchSize']
epochs = config['Epochs']
learning_rate = config['LearningRate']
sequence_length = config['SequenceLength']


class AttentionThree(nn.Module):
    def __init__(self, d_x, n):
        super(AttentionThree, self).__init__()
        self.d_x, self.n = d_x, n
        self.K = nn.Linear(n, d_x)
        self.Q = nn.Linear(n, d_x)
        self.V = nn.Linear(n, d_x)
        self.W_O = nn.Linear(d_x, n)

        self.layer_normlization_1 = nn.LayerNorm(batch_size, sequence_length, self.n)
        self.layer_normlization_2 = nn.LayerNorm(batch_size, sequence_length, self.n)

        self.feed_forward = nn.Linear(self.n, self.n)
        self.activation = nn.ReLU()

    def forward(self, x_orig):
        assert x_orig.shape == (batch_size, sequence_length, self.n)

        keys, queries, values = self.K(x_orig), self.Q(x_orig), self.V(x_orig)
        new_x = nn.Softmax()((keys @ queries.T) / np.sqrt(self.d_x)) @ values
        final_attention_x = self.W_O(new_x) + x_orig
        normalized_x = self.layer_normlization(final_attention_x)

        assert normalized_x.shape == (batch_size, sequence_length, self.n)

        final_x = self.activation(self.feed_forward(normalized_x)) + normalized_x
        final_x_normalized = self.layer_normlization_2(final_x)

        return final_x_normalized

    def num_of_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class AttentionTwo(nn.Module):
    def __init__(self, d_x, n):
        super(AttentionTwo, self).__init__()
        self.d_x, self.n = d_x, n
        self.K = nn.Linear(n, d_x)
        self.Q = nn.Linear(n, d_x)
        self.W_O = nn.Linear(d_x, n)

        self.layer_normlization_1 = nn.LayerNorm(batch_size, sequence_length, self.n)
        self.layer_normlization_2 = nn.LayerNorm(batch_size, sequence_length, self.n)

        self.feed_forward = nn.Linear(self.n, self.n)
        self.activation = nn.ReLU()

    def forward(self, x_orig):
        assert x_orig.shape == (batch_size, sequence_length, self.n)

        keys, queries = self.K(x_orig), self.Q(x_orig)
        new_x = nn.Softmax()((keys @ queries.T) / np.sqrt(self.d_x))
        final_attention_x = self.W_O(new_x) + x_orig
        normalized_x = self.layer_normlization(final_attention_x)

        assert normalized_x.shape == (batch_size, sequence_length, self.n)

        final_x = self.activation(self.feed_forward(normalized_x)) + normalized_x
        final_x_normalized = self.layer_normlization_2(final_x)

        return final_x_normalized

    def num_of_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class Model3(nn.Module):
    def __init__(self, p, d_x, n):
        super(Model3, self).__init__()
        self.network = nn.Sequential(
            *[AttentionThree(d_x, n) for _ in range(p)]
        )

    def forward(self, x):
        y = self.network(x)
        return y

    def num_of_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class Model2(nn.Module):
    def __init__(self, p, d_x, n):
        super(Model2, self).__init__()
        self.network = nn.Sequential(
            *[AttentionTwo(d_x, n) for _ in range(p)]
        )

    def forward(self, x):
        y = self.network(x)
        return y

    def num_of_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params






