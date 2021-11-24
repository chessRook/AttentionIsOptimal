import torch
import models

import torchvision

import torch.nn as nn
import pandas as pd

import torch.optim as optim

import torchvision.transforms as T

######################################
from commons import *
from tqdm import tqdm
from pathlib import Path
from config import Config
from enum import Enum
from models import ResnetModel, LSTModel
from transformer_model import TransformerModel
from transformers_factory import TransformersFactory


######################################


class DataKinds(Enum):
    Test = 'Test'
    Train = 'Train'
    Eval = 'Eval'


class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = Path('./Data/train.csv') if data_path is None else Path(data_path)
        self.data_kind = self.data_kind_extractor(self.data_path)
        self.data = pd.read_csv(self.data_path)
        self.df = pd.DataFrame(self.data)
        self.values_translator = dict()
        self.label_class_name = 'SalePrice'
        self.device = 'cuda:0'
        self.expected_len = 80

    @staticmethod
    def data_kind_extractor(data_path):
        filename = data_path.stem
        data_kind = filename.title()
        data_kind_obj = DataKinds(data_kind)
        return data_kind_obj

    def translate_value(self, class_name, value):
        # Value-Updater
        if class_name not in self.values_translator:
            self.values_translator[class_name] = {value: 1.0, '_MAX_': 1.0}
        else:
            if value not in self.values_translator[class_name]:
                self.values_translator[class_name]['_MAX_'] = self.values_translator[class_name]['_MAX_'] + 1
                self.values_translator[class_name][value] = self.values_translator[class_name]['_MAX_']

        return self.values_translator[class_name][value]

    def training_generator(self):
        for row_idx, row in self.df.iterrows():
            if row_idx == 0:
                continue
            values, label = [], None
            row_as_dct = row.to_dict()
            for class_name, value in row_as_dct.items():
                if class_name == self.label_class_name:
                    label = value
                    continue
                elif not isinstance(value, str) and (value is not None and not pd.isna(value)):
                    to_append_value = float(value)
                elif value is None or pd.isna(value):
                    to_append_value = 0.0
                else:
                    to_append_value = self.translate_value(class_name, value)

                # appending
                values.append(to_append_value)

            if len(values) != self.expected_len:
                continue

            yield values, label

    def train_tensors_generator(self):
        for values, label in self.training_generator():
            values_as_tensor = torch.tensor(values)
            label_as_tensor = torch.tensor(label)
            yield values_as_tensor, label_as_tensor

    def train_tensors_batches_generator(self, batch_size):
        iter_ = self.train_tensors_generator()
        stopper = False
        while not stopper:
            values_batch, label_batch = [], []
            for idx in range(batch_size):
                try:
                    values, label = iter_.__next__()
                    values_batch.append(values)
                    label_batch.append(label)
                except StopIteration:
                    stopper = True
                    break

            if stopper:
                continue

            label_batch_ten = torch.stack(label_batch)
            values_batch_ten = torch.stack(values_batch, dim=0)

            label_batch_ten = label_batch_ten.cuda()
            values_batch_ten = values_batch_ten.cuda()

            yield values_batch_ten, label_batch_ten

    def train_tensors_epochs_generator(self, epochs, batch_size):
        for epoch in range(epochs):
            for inputs_ten, labels_ten in self.train_tensors_batches_generator(batch_size):
                yield inputs_ten, labels_ten


class Runner:
    def __init__(self, budget, seq_len, input_dim, embedding_dim):
        self.train_data, self.eval_data = DataLoader(data_path='./Data/train.csv'), \
                                          DataLoader(data_path='Data/eval.csv')
        self.all_models = self.model_creator(budget, seq_len, input_dim, embedding_dim)
        self.seq_len = seq_len
        self.optimizer = None
        self.a, self.b = int(1e6), int(2e5)

    @staticmethod
    def model_creator(budget, seq_len, input_dim, embedding_dim):
        transformers = TransformersFactory(budget, seq_len, input_dim, embedding_dim)
        return transformers.cuda()

    def runner(self):
        # TODO train different models in parallel
        for model in self.all_models.models_list():
            self.training_setup(model)
            for epoch in range(epochs):
                self.trainer(model, self.train_data, epochs_=1)
                self.trainer(model, self.eval_data, epochs_=1)

        self.cleanup()

    def training_setup(self, model):
        self.optimizer = optim.Adam(model.network.parameters(), lr=learning_rate)

    def trainer(self, model, data_obj, epochs_):
        data_kind = data_obj.data_kind
        for inputs_ten, gold_labels_ten in tqdm(data_obj.train_tensors_epochs_generator(epochs=epochs_,
                                                                                        batch_size=batch_size)):
            self.one_batch_er(model, inputs_ten, gold_labels_ten, data_kind)

    def one_batch_er(self, model, inputs_ten, gold_labels_ten, data_kind):
        model.zero_grad()
        inputs_ten_ = inputs_ten.float()
        pred_labels = model(inputs_ten_)
        pred_labels_chosen = pred_labels[:, 0, 0]
        pred_labels_normalized = self.a * pred_labels_chosen - self.b
        loss_ = self.criterion(gold_labels_ten, pred_labels_normalized)

        self.training_cases_handler(model, data_kind, loss_)

        ###########################################

        self.logger(model, loss_, gold_labels_ten, pred_labels_normalized, data_kind)

        ##########################################

    def training_cases_handler(self, model, data_kind, loss_):
        if data_kind == DataKinds.Train:
            loss_.backward()
            self.optimizer.step()
        elif data_kind in {DataKinds.Eval, DataKinds.Test}:
            model.zero_grad()
        else:
            raise NotImplementedError("This Kind Of Training Isn't Supported Yet")

    @staticmethod
    def logger(model, loss_, gold_labels_ten, pred_labels, data_kind):
        # Error
        run[f"ErrorPercent{model.model_kind}{data_kind.name}"].log(loss_.item())

        # Preparing
        gold = gold_labels_ten.squeeze()[0].item()
        pred = pred_labels.squeeze()[0].item()
        abs_diff = abs(gold - pred)

        # Logging
        run['gold'].log(gold)
        run['pred'].log(pred)
        run["AbsDiff"].log(abs_diff)

    @staticmethod
    def criterion(gold_labels_ten, pred_labels):
        loss_ = (1. - (pred_labels / gold_labels_ten).mean()).abs()
        return loss_

    @staticmethod
    def cleanup():
        run.stop()

    @staticmethod
    def input_shaper(inputs_ten):
        # Init
        inputs_ten = inputs_ten.cuda()
        zeros_ten = torch.zeros(20, 224 - inputs_ten.shape[-1]).cuda()
        # Reshaping
        inp_1 = torch.cat([inputs_ten, zeros_ten], dim=-1)  # 20, 224
        inp_2 = inp_1.unsqueeze(1)
        inp_3 = torch.cat([inp_2, ] * 224, dim=1)
        inp_4 = inp_3.unsqueeze(1)
        inp_5 = torch.cat([inp_4, inp_4, inp_4], dim=1)
        return inp_5


if __name__ == '__main__':
    runner_obj = Runner(int(5_000), int(1_00), int(10), int(3))  # (budget, input_dim, embedding_dim)
    runner_obj.runner()
