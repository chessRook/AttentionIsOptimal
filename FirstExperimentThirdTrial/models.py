import torch
import torchvision

import torch.nn as nn
import pandas as pd

import torch.optim as optim
import neptune.new as neptune

import torchvision.transforms as T

######################################

from tqdm import tqdm
from pathlib import Path
from config import Config
from enum import Enum
from commons import *


###########################################################

class ModelsCommons(nn.Module):
    def __init__(self):
        super(ModelsCommons, self).__init__()
        self.price_normalization = 2_000_000.
        self.price_const_bias = 8_20_000.

    def normalizer(self, x):
        final_price = self.price_normalization * x - self.price_const_bias
        return final_price


###########################################################


class ResnetModel(ModelsCommons):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.price_normalization = 2_000_000.
        self.price_const_bias = 8_20_000.
        self.expected_resolution = 1
        self.transform = T.Compose([T.Normalize((.5, .5, .5), (.5, .5, .5))])
        self.network = torchvision.models.resnet50(pretrained=False)
        self.change_last_layer_resolution(self.network, self.expected_resolution)

    def input_shaper(self, inputs_ten):
        inputs_ten_shaped = self.input_shaper(inputs_ten) / inputs_ten.max()
        return inputs_ten_shaped

    def forward(self, input_):
        out_0 = self.input_shaper(input_)
        out_1 = self.network(self.transform(out_0))
        out_2 = nn.Sigmoid()(out_1)
        # Start From The Mean
        out_3 = self.price_normalization * out_2 - self.price_const_bias
        return out_3

    @staticmethod
    def change_last_layer_resolution(network, expected_resolution):
        network.fc = nn.Linear(2048, expected_resolution)
        torch.nn.init.normal_(network.fc.weight, mean=0, std=.0001)


###########################################################

class LSTModel(ModelsCommons):
    def __init__(self):
        super(LSTModel, self).__init__()
        self.bidirectional = True
        self.input_dim = 10
        self.sequence_length, self.hidden_size, self.batch_size = 1_00, 1_00, batch_size
        self.D, self.L, self.num_layers = (2 if self.bidirectional else 1), self.sequence_length, 5
        self.network = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               bidirectional=True)
        self.h_0 = torch.randn(self.D * self.num_layers, self.batch_size, self.hidden_size).cuda()
        self.c_0 = torch.randn(self.D * self.num_layers, self.batch_size, self.hidden_size).cuda()

    def forward(self, batch_vectors):
        input_ = self.re_shaper(batch_vectors)

        # Checking Input Shape
        assert input_.shape == (self.L, self.batch_size, self.input_dim)

        output_, (h_n, c_n) = self.network(input_, (self.h_0, self.c_0))

        # Checking Output Shape
        assert output_.shape == (self.L, self.batch_size, self.D * self.hidden_size)

        predictions = nn.Sigmoid()(output_[-1, :, -1])
        predictions_normalized = self.normalizer(predictions)

        return predictions_normalized

    def gaps_filler(self, batch_vectors):
        filler = torch.zeros(batch_vectors.shape[0], self.L - batch_vectors.shape[1], device=device)
        filled = torch.cat([batch_vectors, filler], dim=1).cuda()
        shaped = torch.cat([filled.unsqueeze(-1), ] * self.input_dim, dim=-1).cuda()
        return shaped

    def re_shaper(self, batch_vectors):
        batch_shaped_0 = self.gaps_filler(batch_vectors)

        assert batch_shaped_0.shape == (self.batch_size, self.L, self.input_dim)

        batch_shaped = batch_shaped_0.transpose(0, 1)

        assert batch_shaped.shape == (self.L, self.batch_size, self.input_dim)

        return batch_shaped

###########################################################
