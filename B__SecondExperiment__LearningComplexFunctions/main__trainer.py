import torch
import utils
import torchvision
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from commons import *
from utils import limit_iterations
from complex_functions_creator import ComplexFunction
from transformers_factory import DegreeNetworksFactory


class Trainer:
    # TODO replace params with config
    def __init__(self, domain_dim, budget, training_time=int(1e6), embed_shrinking_factor=4):
        self.inp_dim = domain_dim
        self.embed_shrinking_factor = embed_shrinking_factor
        self.learned_function = ComplexFunction(self.inp_dim)
        self.data_generator = self.learned_function.data_generator()
        self.models = DegreeNetworksFactory(budget=budget, input_dim=self.inp_dim,
                                            embedding_dim=int(self.inp_dim / self.embed_shrinking_factor))
        self.training_time = training_time
        self.criterion = self.loss
        self.optimizer = None

    def loss(self, y_p, y_g):
        l1_loss = (y_p - y_g).abs().mean()
        return l1_loss

    def models_trainer(self):
        for model in self.models.models_list():
            self.adapt_optimizer(model)
            self.model_trainer(model)

    def adapt_optimizer(self, model):
        self.optimizer = optim.Adam(model.parameters(), lr=.001)

    def model_trainer(self, model):
        with tqdm()
            for x, y_g in limit_iterations(self.data_generator, limit=self.trainign_time):
                self.optimizer.zero_grad()
                y_p = model(x)
                l = self.criterion(y_p, y_g)
                l.backward()
                self.optimizer.step()

                self.logger(y_g, y_p, l)

    def logger(self, y_g, y_p, l):
        y_g, y_p, l = y_g.item(), y_p.item(), l.item()
        neptune_logger['y_g'].log(y_g)
        neptune_logger['y_p'].log(y_p)
        neptune_logger['l1_loss'].log(l)
