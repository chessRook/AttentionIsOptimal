import random
import torch
import torchvision
import torch.nn as nn

import numpy as np
from random import random as rnd


class ComplexFunction:
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
        self.Sigma = self.variance_generator()

    @staticmethod
    def one_d_domain_sampler():
        x = random.uniform(0, 1)
        return x

    def vec_sampler(self):
        vec = [self.one_d_domain_sampler() for _ in range(self.inp_dim)]
        return vec

    def variance_generator(self):
        matrix = torch.randn(self.inp_dim, self.inp_dim)
        psd_matrix = matrix @ matrix.T
        pd_matrix = torch.eye(self.inp_dim, self.inp_dim) + psd_matrix
        matrix_sum = pd_matrix.abs().sum()
        normalized_pd_matrix = pd_matrix / matrix_sum
        return normalized_pd_matrix.cuda()

    def complex_function(self, x):
        x_ten = torch.tensor(x).cuda()
        res_1 = torch.matmul(self.Sigma, x_ten)
        res_2 = torch.matmul(res_1, x)
        assert res_2.shape == (1,)
        return res_2

    def data_generator(self):
        while True:
            x = self.vec_sampler()
            y = self.complex_function(x)
            yield x, y
