import torch
import torch.nn as nn

def get_activation_cls(nonlinearity):
    if nonlinearity == "relu":
        activation_cls = nn.ReLU
    elif nonlinearity == "sigmoid":
        activation_cls = nn.Sigmoid
    elif nonlinearity == "tanh":
        activation_cls = nn.Tanh
    elif nonlinearity == "leaky_relu":
        activation_cls = nn.LeakyReLU
    elif nonlinearity == "gelu":
        activation_cls = nn.GELU
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
    return activation_cls