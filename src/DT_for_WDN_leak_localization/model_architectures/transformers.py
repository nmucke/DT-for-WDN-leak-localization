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


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input

        self.softmax = nn.Softmax(dim=-1)

        # Embedding dimension of model is a multiple of number of heads

        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. To split into number of heads
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesn't saturate
        Q = Q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))  # (bs, n_heads, q_length, k_length)

        A = self.softmax(scores)  # (bs, n_heads, q_length, k_length)

        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q), batch_size)
        K = self.split_heads(self.W_k(X_k), batch_size)
        V = self.split_heads(self.W_v(X_v), batch_size)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)  # (bs, q_length, dim)

        # Final linear layer
        H = self.W_h(H_cat)  # (bs, q_length, dim)
        return H, A


class EmbedCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.k1convL1 = nn.Linear(input_dim, hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.k1convL1 = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=1
        )
        self.k1convL2 = nn.Conv1d(
            hidden_dim,
            output_dim,
            kernel_size=1
        )
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            input_embed_dim,
            output_embed_dim,
            num_heads,
            embed_hidden_dim,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(input_embed_dim, num_heads, p)
        self.embed_cnn = EmbedCNN(
            input_dim=input_embed_dim,
            output_dim=output_embed_dim,
            hidden_dim=embed_hidden_dim
        )

        self.cnn = CNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=output_dim
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=input_embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=output_embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=output_embed_dim, eps=1e-6)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, input_seq_len, input_embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Compute accross embedding dimension
        x = self.embed_cnn(x)  # (batch_size, input_seq_len, output_embed_dim)

        x = self.activation(x)

        x = self.layernorm2(x)

        # Compute accross sequence dimension
        x = self.cnn(x)  # (batch_size, output_seq_len, output_embed_dim)

        x = self.activation(x)

        x = self.layernorm3(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            input_embed_dim,
            output_embed_dim,
            num_heads,
            embed_hidden_dim,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(input_embed_dim, num_heads, p)
        self.embed_cnn = EmbedCNN(
            input_dim=input_embed_dim,
            output_dim=output_embed_dim,
            hidden_dim=embed_hidden_dim
        )

        self.cnn = CNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=output_dim
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=input_embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=input_embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=output_embed_dim, eps=1e-6)
        self.layernorm4 = nn.LayerNorm(normalized_shape=output_embed_dim, eps=1e-6)

    def forward(self, x, encoder_output):

        # Multi-head self attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, input_seq_len, input_embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Multi-head cross attention
        attn_output, _ = self.mha(X_q=x, X_k=encoder_output, X_v=encoder_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm2(x + attn_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Compute accross embedding dimension
        x = self.embed_cnn(x)  # (batch_size, input_seq_len, output_embed_dim)

        x = self.activation(x)

        x = self.layernorm3(x)

        # Compute accross sequence dimension
        x = self.cnn(x)  # (batch_size, output_seq_len, output_embed_dim)

        x = self.activation(x)

        x = self.layernorm4(x)

        return x

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int=32,
        state_dim: int=128,
        embed_dims: list=[8, 8],
        hidden_neurons: list=[32, 16],
        num_heads: int=2,
    ):
        super().__init__()

        self.hidden_neurons = [state_dim] + hidden_neurons
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.embed_dims = embed_dims


        self.input_layer = nn.Linear(1, self.embed_dims[0])

        self.positional_embedding = PositionalEmbedding(
            dim= self.embed_dims[0]
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                input_dim=self.hidden_neurons[i],
                output_dim=self.hidden_neurons[i+1],
                input_embed_dim=self.embed_dims[i],
                output_embed_dim=self.embed_dims[i+1],
                num_heads=num_heads,
                embed_hidden_dim=embed_dims[i],
            ) for i in range(len(self.hidden_neurons)-1)
        ])

        self.output_layer_1 = nn.Linear(
                self.embed_dims[-1]*self.hidden_neurons[-1],
                self.latent_dim
        )

        self.flatten = nn.Flatten()
    def forward(self, x):

        x = x.unsqueeze(-1)
        x = self.input_layer(x)

        x = self.positional_embedding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.flatten(x)

        x = self.output_layer_1(x)
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int=32,
            state_dim: int=128,
            embed_dims: list=[8, 8],
            hidden_neurons: list=[16, 32],
            pars_dims: list=[119, 24],
            num_heads: int=2,
    ) -> None:
        super().__init__()

        pars_hidden_neurons = [latent_dim] + hidden_neurons
        pars_embed_dims = [embed_dims[0]] + embed_dims

        self.hidden_neurons = [latent_dim] + hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dim = pars_dims
        self.state_dim = state_dim

        self.embed_dims = embed_dims

        if len(pars_dims) == 1:
            pars_embedding_dim = [latent_dim]
        elif len(pars_dims) == 2:
            pars_embedding_dim = [latent_dim, latent_dim]

        total_pars_embedding_dim = sum(pars_embedding_dim)

        self.pars_embedding_layers = nn.ModuleList()
        for i in range(len(self.pars_dim)):
            self.pars_embedding_layers.append(
                nn.Embedding(
                    num_embeddings=pars_dims[i],
                    embedding_dim=pars_embedding_dim[i]
                )
            )

        self.pars_layer_in = nn.Linear(
                in_features=total_pars_embedding_dim,
                out_features=(self.hidden_neurons[0] * self.embed_dims[0])
        )

        self.input_layer = nn.Linear(1, self.embed_dims[0])

        self.positional_embedding = PositionalEmbedding(
            dim=self.embed_dims[0]
        )

        self.pars_encoder_layers = nn.ModuleList([
            EncoderLayer(
                input_dim=pars_hidden_neurons[i],
                output_dim=pars_hidden_neurons[i+1],
                input_embed_dim=pars_embed_dims[i],
                output_embed_dim=pars_embed_dims[i+1],
                num_heads=num_heads,
                embed_hidden_dim=pars_embed_dims[i],
            ) for i in range(len(pars_hidden_neurons)-1)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                input_dim=self.hidden_neurons[i],
                output_dim=self.hidden_neurons[i+1],
                input_embed_dim=self.embed_dims[i],
                output_embed_dim=self.embed_dims[i+1],
                num_heads=num_heads,
                embed_hidden_dim=embed_dims[i],
            ) for i in range(len(self.hidden_neurons)-1)
        ])


        self.flatten = nn.Flatten()

        self.output_layer_1 = nn.Linear(
                self.embed_dims[-1]*self.hidden_neurons[-1],
                self.state_dim
        )


    def forward(self, x, pars):


        pars = [emb_layer(pars[:, i])
            for i, emb_layer in enumerate(self.pars_embedding_layers)]
        pars = torch.cat(pars, 1)

        pars = self.pars_layer_in(pars)
        pars = pars.view(-1, self.hidden_neurons[0], self.embed_dims[0])
        x = x.unsqueeze(-1)
        x = self.input_layer(x)

        x = self.positional_embedding(x)

        for layer, pars_layer in zip(self.decoder_layers, self.pars_encoder_layers):
            pars = pars_layer(pars)

            x = layer(x, pars)

        x = self.flatten(x)
        x = self.output_layer_1(x)

        return x



class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int=32,
            state_dim: int=128,
            embed_dims: list=[8, 8],
            hidden_neurons: list=[16, 32],
            num_heads: int=2,
    ) -> None:
        super().__init__()

        self.hidden_neurons = [latent_dim] + hidden_neurons
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.embed_dims = embed_dims


        self.input_layer = nn.Linear(1, self.embed_dims[0])

        self.positional_embedding = PositionalEmbedding(
            dim=self.embed_dims[0]
        )

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                input_dim=self.hidden_neurons[i],
                output_dim=self.hidden_neurons[i+1],
                input_embed_dim=self.embed_dims[i],
                output_embed_dim=self.embed_dims[i+1],
                num_heads=num_heads,
                embed_hidden_dim=embed_dims[i],
            ) for i in range(len(self.hidden_neurons)-1)
        ])


        self.flatten = nn.Flatten()

        self.output_layer_1 = nn.Linear(
                self.embed_dims[-1]*self.hidden_neurons[-1],
                self.state_dim
        )


    def forward(self, x, ):

        x = x.unsqueeze(-1)
        x = self.input_layer(x)

        x = self.positional_embedding(x)

        for layer in self.decoder_layers:

            x = layer(x, x)

        x = self.flatten(x)
        x = self.output_layer_1(x)

        return x
