import numpy as np
import torch
from torch import nn

def create_model():
    modelNN = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )

    return modelNN

def count_parameters(model):
    res = 0
    for param in model.parameters():
      x = 1
      for s in param.shape:
        x *= s
      res += x
    return res
