import pdb
import torch
import torch.nn as nn
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



class Encoder(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        hidden_neurons: list, 
        embed_dim: int,
        latent_dim: int,
        transformer=True
        ):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.transformer = transformer


        self.initial_state_encoder = nn.Sequential(
            nn.Unflatten(1, (state_dim, 1)),
            nn.Linear(
                in_features=1,
                out_features=embed_dim,
            ),
            nn.LeakyReLU(),
        )

        self.initial_positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            seq_len=state_dim
        )

        self.initial_transformer_layers = nn.ModuleList(
            [EncoderLayer(
                embed_dim=embed_dim,
                num_heads=2,
                embed_hidden_dim=embed_dim,
                p=0.1
            ) for _ in range(1)]
        )


        '''
        self.reduce_state_embed_dim = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim//2,
            bias=True
        )
        '''

        self.flatten = nn.Flatten()

        hidden_neurons = [state_dim*embed_dim] + hidden_neurons
        #hidden_neurons = [state_dim*(embed_dim//2)] + hidden_neurons
        #hidden_neurons = [state_dim] + hidden_neurons
        self.dim_reduction_layers = nn.ModuleList(
            [DimReductionLayer(
                in_features=hidden_neurons[i],
                out_features=hidden_neurons[i+1]
            )
             for i in range(len(hidden_neurons)-1)]
        )
        
        self.final_dim_reduction_layer = DimReductionLayer(
                in_features=hidden_neurons[-1],
                out_features=latent_dim
            )
        
        self.unflatten = nn.Unflatten(1, (latent_dim, 1))
        self.final_embed_dim_increase = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )

        self.final_positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            seq_len=latent_dim
        )

        self.final_state_transformer_layers = nn.ModuleList(
            [EncoderLayer(
                embed_dim=embed_dim,
                num_heads=2,
                embed_hidden_dim=embed_dim,
                p=0.1
            ) for _ in range(2)]
        )
        '''
        self.reduce_latent_embed_dim = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim//2,
            bias=True
        )
        '''

        self.final_layer = nn.Linear(
            in_features=latent_dim*embed_dim,#(embed_dim//2),
            out_features=latent_dim
        )
    
    def forward(
        self, 
        states: torch.Tensor, 
        ) -> torch.Tensor:

        states = self.initial_state_encoder(states)

        states = self.initial_positional_embedding(states)

        for layer in self.initial_transformer_layers:
            states = layer(states)

        #states = self.reduce_state_embed_dim(states)
        latent_state = self.flatten(states)

        for layer in self.dim_reduction_layers:
            latent_state = layer(latent_state)

        latent_state = self.final_dim_reduction_layer(latent_state)

        latent_state = self.unflatten(latent_state)
        latent_state = self.final_embed_dim_increase(latent_state)
        latent_state = self.final_positional_embedding(latent_state)

        for layer in self.final_state_transformer_layers:
            latent_state = layer(latent_state)

        #latent_state = self.reduce_latent_embed_dim(latent_state)
        latent_state = self.flatten(latent_state)
        latent_state = self.final_layer(latent_state)

        return latent_state






