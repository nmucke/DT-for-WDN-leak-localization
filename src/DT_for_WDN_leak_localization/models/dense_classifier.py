import pdb
import torch
import torch.nn as nn
from DT_for_WDN_leak_localization.model_architectures.dense import DenseResNetLayer
from DT_for_WDN_leak_localization.model_architectures.transformers import DecoderLayer, EncoderLayer, PositionalEmbedding

from DT_for_WDN_leak_localization.models.parameter_encoder import (
    CategoricalEmbeddingLayer,
    ParameterEncoder,
    TransformerEmbedding
)

class DimReductionLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int
        ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            ),
            nn.LeakyReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class DenseClassifier(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        hidden_neurons: list,
        pars_dim: int,
        ):
        super(DenseClassifier, self).__init__()

        self.state_dim = state_dim
        self.hidden_neurons = hidden_neurons
        self.pars_dim = pars_dim

        self.softmax = nn.Softmax(dim=1)

        hidden_neurons = [state_dim] + hidden_neurons

        self.res_net_layers = nn.ModuleList(
            [DenseResNetLayer(
                in_features=hidden_neurons[i],
                out_features=hidden_neurons[i]
            ) for i in range(len(hidden_neurons))]
        )

        self.dim_reduction_layers = nn.ModuleList(
            [DimReductionLayer(
                in_features=hidden_neurons[i],
                out_features=hidden_neurons[i+1]
            ) for i in range(len(hidden_neurons)-1)]
        )

        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(
                num_features=hidden_neurons[i+1]
            ) for i in range(len(hidden_neurons)-1)]
        )

        self.output_layer = nn.Linear(
            in_features=hidden_neurons[-1],
            out_features=pars_dim,
            bias=False
        )
    
    def forward(
        self, 
        states: torch.Tensor, 
        ) -> torch.Tensor:

        for i in range(len(self.hidden_neurons)):
            states = self.res_net_layers[i](states)
            states = self.dim_reduction_layers[i](states)
            states = self.batch_norm_layers[i](states)

        states = self.output_layer(states)

        states = self.softmax(states)

        return states






