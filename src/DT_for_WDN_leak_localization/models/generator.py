import pdb
import torch
import torch.nn as nn
from DT_for_WDN_leak_localization.model_architectures.transformers import DecoderLayer, EncoderLayer, PositionalEmbedding

from DT_for_WDN_leak_localization.models.parameter_encoder import (
    CategoricalEmbeddingLayer,
    ParameterEncoder,
    TransformerEmbedding
)

class DimIncreaseLayer(nn.Module):

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

class Generator(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        hidden_neurons: list, 
        embed_dim: int,
        latent_dim: int,
        pars_dims: list,
        transformer=True
        ):
        super(Generator, self).__init__()

        self.state_dim = state_dim
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dims = pars_dims
        self.transformer = transformer

        self.initial_pars_encoder = ParameterEncoder(
            embed_dim=embed_dim,
            seq_len=latent_dim,
            pars_dims=pars_dims,
            num_layers=2,
            transformer=True
        )

        self.initial_state_encoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1)),
            nn.Linear(
                in_features=1,
                out_features=embed_dim,
            ),
            nn.LeakyReLU(),
        )

        self.initial_positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            seq_len=latent_dim
        )

        self.initial_transformer_layers = nn.ModuleList(
            [DecoderLayer(
                embed_dim=embed_dim,
                num_heads=2,
                embed_hidden_dim=embed_dim,
                p=0.1
            ) for _ in range(2)]
        )

        self.flatten = nn.Flatten()

        hidden_neurons = [latent_dim*embed_dim] + hidden_neurons
        self.dim_reduction_layers = nn.ModuleList(
            [DimIncreaseLayer(
                in_features=hidden_neurons[i],
                out_features=hidden_neurons[i+1]
            )
             for i in range(len(hidden_neurons)-1)]
        )
        
        self.final_dim_reduction_layer = DimIncreaseLayer(
                in_features=hidden_neurons[-1],
                out_features=state_dim
            )
        
        self.unflatten = nn.Unflatten(1, (state_dim, 1))
        self.final_embed_dim_increase = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )

        self.final_positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            seq_len=state_dim
        )

        self.final_pars_encoder = ParameterEncoder(
            embed_dim=embed_dim,
            seq_len=state_dim,
            pars_dims=pars_dims,
            num_layers=2,
            transformer=True
        )

        self.final_transformer_layers = nn.ModuleList(
            [DecoderLayer(
                embed_dim=embed_dim,
                num_heads=2,
                embed_hidden_dim=embed_dim,
                p=0.1
            ) for _ in range(1)]
        )

        self.final_layer = nn.Linear(
            in_features=embed_dim*state_dim,
            out_features=state_dim
        )
    
    def forward(
        self, 
        latent_state: torch.Tensor, 
        pars: torch.Tensor
        ) -> torch.Tensor:

        pars_attn = self.initial_pars_encoder(pars)

        latent_state = self.initial_state_encoder(latent_state)

        latent_state = self.initial_positional_embedding(latent_state)
        for layer in self.initial_transformer_layers:
            latent_state = layer(latent_state, pars_attn)

        latent_state = self.flatten(latent_state)

        for layer in self.dim_reduction_layers:
            latent_state = layer(latent_state)

        latent_state = self.final_dim_reduction_layer(latent_state)

        latent_state = self.unflatten(latent_state)
        latent_state = self.final_embed_dim_increase(latent_state)
        latent_state = self.final_positional_embedding(latent_state)

        pars_attn = self.final_pars_encoder(pars)
        for layer in self.final_transformer_layers:
            latent_state = layer(latent_state, pars_attn)

        latent_state = self.flatten(latent_state)
        latent_state = self.final_layer(latent_state)

        return latent_state






