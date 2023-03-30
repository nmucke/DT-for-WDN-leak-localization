import pdb
import torch
import torch.nn as nn
from DT_for_WDN_leak_localization.model_architectures.dense import DenseResNetLayer
from DT_for_WDN_leak_localization.model_architectures.transformers import (
    DecoderLayer, 
    EncoderLayer, 
    PositionalEmbedding
)

from DT_for_WDN_leak_localization.models.parameter_encoder import (
    ParameterEncoder,
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

class Decoder(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        hidden_neurons: list, 
        latent_dim: int,
        pars_dims: list,
        ):
        super(Decoder, self).__init__()

        self.state_dim = state_dim
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dims = pars_dims

        self.transformer = False

        hidden_neurons = [latent_dim + latent_dim] + hidden_neurons

        self.initial_pars_encoder = ParameterEncoder(
            embed_dim=latent_dim,
            seq_len=latent_dim,
            pars_dims=pars_dims,
            num_layers=2,
            transformer=False
        )

        self.init_batch_norm = nn.BatchNorm1d(
            num_features=hidden_neurons[0]
        )
        

        self.res_net_layers = nn.ModuleList(
            [DenseResNetLayer(
                in_features=hidden_neurons[i],
                out_features=hidden_neurons[i]
            ) for i in range(len(hidden_neurons)-1)]
        )

        self.dim_increase_layers = nn.ModuleList(
            [DimIncreaseLayer(
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
            out_features=state_dim
        )

    def pars_forward(self, pars: torch.Tensor) -> torch.Tensor:

        pars_attn_1 = self.initial_pars_encoder(pars)

        return pars_attn_1, torch.tensor([0])
    
    def state_forward(
        self, 
        latent_state: torch.Tensor,
        pars_attn_1: torch.Tensor,
        pars_attn_2: torch.Tensor = None
        ) -> torch.Tensor:
        
        state = torch.cat([latent_state, pars_attn_1], dim=1)

        state = self.init_batch_norm(state)


        for i in range(len(self.res_net_layers)):
            state = self.res_net_layers[i](state)
            state = self.dim_increase_layers[i](state)
            state = self.batch_norm_layers[i](state)
            
        state = self.output_layer(state)

        return state

    def forward(
        self, 
        latent_state: torch.Tensor, 
        pars: torch.Tensor
        ) -> torch.Tensor:

        pars_attn_1, _ = self.pars_forward(pars)

        state = self.state_forward(
            latent_state=latent_state,
            pars_attn_1=pars_attn_1
        ) 

        return state






