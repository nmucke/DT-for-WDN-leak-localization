import torch
import torch.nn as nn

from DT_for_WDN_leak_localization.model_architectures.transformers import EncoderLayer, PositionalEmbedding

class CategoricalEmbeddingLayer(nn.Module):
    """ Embedding layer for categorical parameters"""
    def __init__(
        self,
        pars_dims: int,
        embed_dim: int
        ):
        super().__init__()

        self.pars_dims = pars_dims
        self.embed_dim = embed_dim

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(
                num_embeddings=pars_dims[i],
                embedding_dim=embed_dim
            ) for i in range(len(pars_dims))]
        )

        self.projection_layer = nn.Linear(
            in_features=len(pars_dims) * embed_dim,
            out_features=embed_dim
        )

    def forward(self, pars: torch.Tensor) -> torch.Tensor:

        pars = [
            emb_layer(pars[:, i]) for i, emb_layer in enumerate(self.embedding_layers)
            ]
        pars = torch.cat(pars, 1)

        pars = self.projection_layer(pars)
        
        return pars

class TransformerEmbedding(nn.Module):
    """ Transformer embedding layer for categorical parameters"""

    def __init__(
        self,
        pars_dims: int,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 2,
        num_layers: int = 1
        ):
        super().__init__()

        self.pars_dims = pars_dims
        self.embed_dim = embed_dim

        self.positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim
        )

        self.transformer_encoder_layers = nn.ModuleList(
            [EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                embed_hidden_dim=embed_dim,
                p=0.1
            ) for i in range(num_layers)]
        )
    
    def forward(self, pars: torch.Tensor) -> torch.Tensor:

        pars = self.positional_embedding(pars)

        for layer in self.transformer_encoder_layers:
            pars = layer(pars)
            
        return pars

class ParameterEncoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        seq_len: int,
        pars_dims: list,
        num_layers: int = 1,
        transformer=True
        ):
        super(ParameterEncoder, self).__init__()

        self.initial_embedding = CategoricalEmbeddingLayer(
            pars_dims=pars_dims,
            embed_dim=embed_dim
        )

        if transformer:
            self.encode = nn.Sequential(
                nn.Unflatten(1, (1, embed_dim)),
                nn.Conv1d(
                    in_channels=1,
                    out_channels=seq_len,
                    kernel_size=1
                ),
                nn.LeakyReLU(),
                TransformerEmbedding(
                    pars_dims=pars_dims,
                    embed_dim=embed_dim,
                    seq_len=seq_len,
                    num_heads=2,
                    num_layers=num_layers
                )
            )
        else:
            self.encode = nn.Sequential(
                nn.Linear(
                    in_features=embed_dim,
                    out_features=embed_dim
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features=embed_dim,
                    out_features=embed_dim
                )
            )

    def forward(self, pars):
        
        pars = self.initial_embedding(pars)
        pars = self.encode(pars)
        
        return pars
