import torch
import torch.nn as nn

from toy_disentanglement.utils import get_activation_cls


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation="relu", output_activation="tanh"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation

        activation_cls = get_activation_cls(activation)

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation_cls())
        output_activation_cls = get_activation_cls(self.output_activation)
        layers.append(output_activation_cls())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @torch.no_grad()
    def get_layer_representation(self, x, idx):
        return self.layers[:idx](x)
    
    @torch.no_grad()
    def get_all_layer_representations(self, x):
        representations = []
        for i in range(1, len(self.layers), 2):  # get representations after each nonlinear activation
            x = self.layers[i-1](x)  # Linear layer
            x = self.layers[i](x)    # Activation layer
            representations.append(x)
        return representations