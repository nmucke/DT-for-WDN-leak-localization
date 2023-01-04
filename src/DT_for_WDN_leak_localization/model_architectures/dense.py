import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time
import math
import torch.nn.functional as F

def normal_init(m, mean, std) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def get_activation_function(activation_function_name: str = 'leaky_relu'):
    if activation_function_name == 'elu':
        activation_function = nn.ELU()
    elif activation_function_name == 'sigmoid':
        activation_function = nn.Sigmoid()
    elif activation_function_name == 'leaky_relu':
        activation_function = nn.LeakyReLU()
    elif activation_function_name == 'tanh':
        activation_function = nn.Tanh()

    return activation_function


class Encoder(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 32, 
        state_dim: int = 128, 
        hidden_neurons: list = [64, 32, 16], 
        activation: str = 'leaky_relu'
        ) -> None:
        super().__init__()

        self.activation = get_activation_function(activation)
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim

        self.dense_in = nn.Linear(
                in_features=state_dim,
                out_features=self.hidden_neurons[0],
                bias=False
        )
        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                    in_features=hidden_neurons[i],
                    out_features=hidden_neurons[i + 1],
                    bias=False
                ) for i in range(len(hidden_neurons)-1)]
        )

        self.dense_out = nn.Linear(in_features=hidden_neurons[-1],
                                   out_features=latent_dim,
                                   bias=False
                                   )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_neurons[i])
                 for i in range(1, len(self.hidden_neurons))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dense_in(x)
        x = self.activation(x)
        #x = self.batch_norm_in(x)

        for dense_layer, batch_norm in zip(
                self.dense_layers,
                self.batch_norm_layers,
        ):
            '''
            for dense_layer in self.dense_layers:
            '''
            x = dense_layer(x)
            x = self.activation(x)
            x = batch_norm(x)


        x = self.dense_out(x)
        return x

class SupervisedDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 32,
            state_dim: int = 128,
            hidden_neurons: list = [16, 32, 64],
            pars_dim: tuple = (119),
            pars_embedding_dim: int = 32,
            activation: str = 'leaky_relu'
    ) -> None:
        super().__init__()

        self.activation = get_activation_function(activation)
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dim = pars_dim

        self.sigmoid = nn.Sigmoid()

        if len(pars_dim) == 1:
            pars_embedding_dim = [pars_embedding_dim]
        elif len(pars_dim) == 2:
            pars_embedding_dim = [pars_embedding_dim, pars_embedding_dim//2]

        total_pars_embedding_dim = sum(pars_embedding_dim)

        self.pars_embedding_layers = nn.ModuleList()
        for i in range(len(self.pars_dim)):
            self.pars_embedding_layers.append(
                nn.Embedding(
                    num_embeddings=pars_dim[i],
                    embedding_dim=pars_embedding_dim[i]
                )
            )

        self.dense_in = nn.Linear(
            in_features=latent_dim,
            out_features=self.hidden_neurons[0],
            bias=False
        )

        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                    in_features=self.hidden_neurons[i] + total_pars_embedding_dim,
                    out_features=self.hidden_neurons[i + 1],
                    bias=False
                ) for i in range(len(hidden_neurons)-1)]
        )

        self.dense_out = nn.Linear(
            in_features=hidden_neurons[-1] + total_pars_embedding_dim,
            out_features=state_dim,
            bias=False
        )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_neurons[i])
                 for i in range(1, len(self.hidden_neurons))]
        )

    def forward(self, x: torch.Tensor, pars: torch.Tensor) -> torch.Tensor:

        x = self.dense_in(x)
        x = self.activation(x)
        x = self.batch_norm_in(x)
        pars = [emb_layer(pars[:, i])
            for i, emb_layer in enumerate(self.pars_embedding_layers)]
        pars = torch.cat(pars, 1)

        for dense_layer, batch_norm in zip(
                self.batch_norm_layers,
        ):

            x = torch.cat((x, pars), dim=1)

            x = dense_layer(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = torch.cat((x, pars), dim=1)

        x = self.dense_out(x)

        return x

class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 32,
            state_dim: int = 128,
            hidden_neurons: list = [16, 32, 64],
            activation: str = 'leaky_relu'
    ) -> None:
        super().__init__()

        self.activation = get_activation_function(activation)
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim

        self.dense_in = nn.Linear(
            in_features=latent_dim,
            out_features=self.hidden_neurons[0],
            bias=False
        )

        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                    in_features=self.hidden_neurons[i],
                    out_features=self.hidden_neurons[i + 1],
                    bias=False
                ) for i in range(len(hidden_neurons)-1)]
        )

        self.dense_out = nn.Linear(
            in_features=hidden_neurons[-1],
            out_features=state_dim,
            bias=False
        )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_neurons[i])
                 for i in range(1, len(self.hidden_neurons))]
        )

    def forward(self, x: torch.Tensor, pars: torch.Tensor) -> torch.Tensor:

        x = self.dense_in(x)
        x = self.activation(x)
        x = self.batch_norm_in(x)

        for dense_layer, batch_norm in zip(
                self.batch_norm_layers,
        ):
            x = dense_layer(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = self.dense_out(x)

        return x