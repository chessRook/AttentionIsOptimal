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


class TransformerModel(nn.Module):
    def __init__(self, attention_class, depth, seq_len, input_dim, embedding_dim):
        super(TransformerModel, self).__init__()
        self.E = 10
        self.depth = depth
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.attention_class = attention_class
        self.before_start_embeder = nn.Linear(1, self.E)
        self.model_kind = ModelKinds.creator(attention_class)
        self.layers_lst_1 = self.before_start_embeder
        self.layers_lst_2 = \
            [self.attention_factory(attention_class, self.seq_len, input_dim, embedding_dim)
             for _ in range(depth)]

        self.network \
            = \
            nn.Sequential(
                *self.layers_lst_2
            )

    def attention_factory(self, attention_class, seq_len, input_dim, embedding_dim):
        if attention_class == 'Attention2':
            attention = Attention2(seq_len, input_dim, embedding_dim)
        elif attention_class == 'Attention3':
            attention = Attention3(seq_len, input_dim, embedding_dim)
        elif attention_class == 'Attention4':
            attention = Attention4(input_dim, embedding_dim)
        else:
            raise NotImplementedError('Only Attention 2-3-4 Are Supported.')

        return attention

    def forward(self, batch):
        batch_shaped = self.shaper(batch)
        value = self.network(batch_shaped)
        return value

    def shaper(self, batch):
        complement = torch.zeros(batch_size, self.seq_len - batch.shape[1]).cuda()
        batch_seq_len_padded = torch.cat([batch, complement], dim=-1)
        with_embed = batch_seq_len_padded.unsqueeze(-1)
        embedded = self.before_start_embeder(with_embed)
        pos_embedded = nn.ELU()(embedded)
        return pos_embedded

    def count_parameters(self):
        total_params = sum(
            param.numel()
            for param in self.network.parameters()
            if param.requires_grad
        )
        return total_params


class AttentionCommons(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(AttentionCommons, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def embedding_layer_creator(self):
        new_layer = nn.Linear(self.input_dim, self.embedding_dim)
        return new_layer

    def shaper(self, x):
        x_1 = x.unsqueeze(0).transpose(0, 1)
        diff_last_dim = self.input_dim - x_1.shape[-1]
        pad_shape = x_1.shape[:-1] + (diff_last_dim,)
        pad = torch.zeros(pad_shape).cuda()
        x_shaped = torch.cat([x_1, pad], dim=-1)
        return x_shaped.cuda()


class Attention4(AttentionCommons):
    def __init__(self, input_dim, embedding_dim):
        super(Attention4, self).__init__(input_dim, embedding_dim)
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.W_1 = self.embedding_layer_creator()
        self.W_2 = self.embedding_layer_creator()
        self.W_3 = self.embedding_layer_creator()
        self.W_4 = self.embedding_layer_creator()
        self.W_O = nn.Linear(embedding_dim, input_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        embed_1, embed_2, embed_3, embed_4 \
            = \
            self.W_1(x), self.W_2(x), self.W_3(x), self.W_4(x)
        attention_1 = embed_1.T @ embed_2
        attention_2 = embed_3.T @ embed_4
        total_attention = attention_1 @ attention_2
        output_1 = self.W_O(total_attention)
        output_2 = self.activation(output_1)

        return output_2


class Attention3(AttentionCommons):
    def __init__(self, seq_len, input_dim, embedding_dim):
        super(Attention3, self).__init__(input_dim, embedding_dim)
        self.seq_len = seq_len
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.W_1 = self.embedding_layer_creator()
        self.W_2 = self.embedding_layer_creator()
        self.W_3 = self.embedding_layer_creator()
        self.W_O = nn.Linear(self.embedding_dim, input_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        embed_1, embed_2, embed_3 = self.W_1(x), self.W_2(x), self.W_3(x)

        attention_1 = embed_1 @ embed_2.transpose(-1, -2)
        attention_2 = attention_1 @ embed_3

        output_1 = self.W_O(attention_2)
        output_2 = self.activation(output_1)

        return output_2


# git commit -m "first commit"
# git remote add origin git@github.com:alexpchin/<reponame>.git
# git push -u origin master

class Attention2(AttentionCommons):
    def __init__(self, seq_len, input_dim, embedding_dim):
        super(Attention2, self).__init__(input_dim, embedding_dim)
        self.seq_len = seq_len
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.W_1 = self.embedding_layer_creator()
        self.W_2 = self.embedding_layer_creator()
        self.W_O = nn.Linear(self.seq_len, input_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        embed_1, embed_2 = self.W_1(x), self.W_2(x)
        attention = embed_1 @ embed_2.transpose(-1, -2)
        output_1 = self.W_O(attention)
        output_2 = self.activation(output_1)

        return output_2
