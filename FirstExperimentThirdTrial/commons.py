import neptune.new as neptune

from enum import Enum
from config import Config

######################################

cfg = Config()
device = 'cuda'
DEBUG_MODE = True
learning_rate = cfg['learningRate']
model_class = cfg['modelType']
epochs = cfg['Epochs']
batch_size = cfg['BatchSize']

######################################
import neptune.new as neptune

run = neptune.init(
    project="chaimchess/AttentionIsOptimal",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBz"
              "Oi8vYXBwLm5lcHR1bmUuYWkiLCJhc"
              "GlfdXJsIjoiaHR0cHM6Ly9hcHAubmV"
              "wdHVuZS5haSIsImFwaV9rZXkiOiI1M"
              "TE2YjA5MC1mNGE0LTQ1ODgtYjQ3MC0x"
              "NTFkOTYwMTg5MWIifQ==",
)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params


########################################


class ModelKinds(Enum):
    Transformer2 = 'Transformer2'
    Transformer3 = 'Transformer3'
    Transformer4 = 'Transformer4'

    @staticmethod
    def creator(attention_class):
        translator = {'Attention2': ModelKinds.Transformer2,
                      'Attention3': ModelKinds.Transformer3,
                      'Attention4': ModelKinds.Transformer4,
                      }
        model_kind = translator[attention_class]
        return model_kind

########################################
