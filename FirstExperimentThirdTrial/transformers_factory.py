import math

import numpy as np

from transformer_model import TransformerModel


class TransformersFactory:
    def __init__(self, budget, seq_len, input_dim, embedding_dim):
        super(TransformersFactory, self).__init__()
        self.budget = budget
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.transformer_2 = self.create_transformer('Attention2')
        self.transformer_3 = self.create_transformer('Attention3')
        self.transformer_4 = self.create_transformer('Attention4')

    def create_transformer(self, attention_class):
        one_layer_budget = self.one_layer_budget(attention_class)
        num_of_layers = self.depth_calculator(one_layer_budget)
        class_transformer = \
            TransformerModel(attention_class, depth=num_of_layers,
                             seq_len=self.seq_len, input_dim=self.input_dim,
                             embedding_dim=self.embedding_dim).cuda()
        return class_transformer

    def depth_calculator(self, one_layer_budget):
        fractional_depth = self.budget / one_layer_budget
        actual_depth = int(np.floor(fractional_depth))
        return actual_depth

    def one_layer_budget(self, attention_class):
        one_layer_model = TransformerModel(attention_class, depth=1,
                                           seq_len=self.seq_len,
                                           input_dim=self.input_dim,
                                           embedding_dim=self.embedding_dim)
        one_layer_budget = one_layer_model.count_parameters()

        return one_layer_budget

    def cuda(self):
        self.transformer_2 = self.transformer_2.cuda()
        self.transformer_3 = self.transformer_3.cuda()
        self.transformer_4 = self.transformer_4.cuda()

        return self

    def models_list(self):
        models_list_ = [self.transformer_2, self.transformer_3, self.transformer_4]
        return models_list_
