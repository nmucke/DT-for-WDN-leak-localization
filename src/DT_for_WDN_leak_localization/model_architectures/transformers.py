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
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
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


class EncoderLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim,
            num_heads,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(embed_dim, num_heads, p)

        self.dropout = nn.Dropout(p)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, seq_len, embed_dim)

        # Adding residual connection
        x = x + self.dropout(attn_output)

        # Layer norm
        x = self.layernorm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.layernorm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim,
            num_heads,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(embed_dim, num_heads, p)

        self.dropout = nn.Dropout(p)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x, encoder_output):

        # Multi-head self attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, input_seq_len, input_embed_dim)

        # Adding residual connection
        x = x + self.dropout(attn_output)

        # Layer norm
        x = self.layernorm1(x)  # (batch_size, input_seq_len, input_embed_dim)

        # Multi-head cross attention
        attn_output, _ = self.mha(X_q=x, X_k=encoder_output, X_v=encoder_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Adding residual connection
        x = x + self.dropout(attn_output)

        # Layer norm
        x = self.layernorm2(x)  # (batch_size, input_seq_len, input_embed_dim)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.layernorm3(x)

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

class EncoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int=128,  
        output_dim: int=128,
        embed_dim: int=8,
        num_heads: int=2,
    ):
        super(EncoderBlock, self).__init__()


        self.embed_increasing_layer = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )
        self.attention_layer = EncoderLayer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        self.embed_reduction_layer = nn.Linear(
            in_features=embed_dim,
            out_features=1
        )
        self.dim_reduction_layer = nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed_increasing_layer(x)
        x = self.attention_layer(x)
        x = self.embed_reduction_layer(x)
        x = x.squeeze(-1)
        x = self.dim_reduction_layer(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int=128,  
        output_dim: int=128,
        embed_dim: int=8,
        num_heads: int=2,
    ):
        super(DecoderBlock, self).__init__()


        self.embed_increasing_layer1 = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )
        self.attention_layer1 = EncoderLayer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        self.embed_reduction_layer1 = nn.Linear(
            in_features=embed_dim,
            out_features=1
        )
        self.dim_increasing_layer1 = nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed_increasing_layer1(x)
        x = self.attention_layer1(x)
        x = self.embed_reduction_layer1(x)
        x = x.squeeze(-1)
        x = self.dim_increasing_layer1(x)
        return x

class CrossAttentionDecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int=128,  
        output_dim: int=128,
        embed_dim: int=8,
        num_heads: int=2,
    ):
        super(CrossAttentionDecoderBlock, self).__init__()


        self.pars_attention_layer1 = EncoderLayer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.embed_increasing_layer1 = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )

        self.cross_attention_layer1 = DecoderLayer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        self.embed_reduction_layer1 = nn.Linear(
            in_features=embed_dim,
            out_features=1
        )
        self.dim_increasing_layer1 = nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )

    def forward(self, x, pars):
        x = x.unsqueeze(-1)

        pars_attn = self.pars_attention_layer1(pars)

        x = self.embed_increasing_layer1(x)
        x = self.cross_attention_layer1(x, pars_attn)
        x = self.embed_reduction_layer1(x)
        x = x.squeeze(-1)
        x = self.dim_increasing_layer1(x)

        return x, pars_attn

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int=32,
        state_dim: int=128,
        embed_dim: int=8,
        hidden_neurons: list=[32, 16],
        num_heads: int=2,
    ):
        super().__init__()

        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.activation = nn.LeakyReLU()

        self.input_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.input_attention = EncoderLayer(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.input_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True
        )

        self.dim_reduction_layers = nn.Sequential(
            nn.Linear(
                in_features=state_dim,
                out_features=hidden_neurons[0],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[0],
                out_features=hidden_neurons[1],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[1],
                out_features=latent_dim,
                bias=True
            ),
            #nn.LeakyReLU(),
        )

        self.output_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.output_attention = EncoderLayer(
            input_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.output_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=False
        )

        self.input_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        self.output_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        '''
        
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(
                input_dim=state_dim,
                output_dim=hidden_neurons[0],
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            EncoderBlock(
                input_dim=hidden_neurons[0],
                output_dim=hidden_neurons[1],
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            EncoderBlock(
                input_dim=hidden_neurons[1],
                output_dim=latent_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        )
        '''

    def forward(self, x):
        #x = self.encoder_blocks(x)

        x = x.unsqueeze(-1)
        x = self.input_increase_embedding(x)
        x = self.activation(x)
        #x = self.input_pos_encoding(x)
        x = self.input_attention(x)
        x = self.input_decrease_embedding(x)
        x = self.activation(x)
        x = x.squeeze(-1)
        
        x = self.dim_reduction_layers(x)

        x = x.unsqueeze(-1)
        x = self.output_increase_embedding(x)
        x = self.activation(x)
        #x = self.output_pos_encoding(x)
        x = self.output_attention(x)
        x = self.output_decrease_embedding(x)
        x = self.activation(x)
        x = x.squeeze(-1)
        return x


class SupervisedDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int=32,
            state_dim: int=128,
            embed_dim: int=8,
            hidden_neurons: list=[16, 32],
            pars_dims: list=[119, 24],
            num_heads: int=2,
    ) -> None:
        super().__init__()

        self.hidden_neurons = [latent_dim] + hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dim = pars_dims
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.activation = nn.LeakyReLU()

        if len(pars_dims) == 1:
            pars_embedding_dim = [latent_dim]
        elif len(pars_dims) == 2:
            pars_embedding_dim = [latent_dim//2, latent_dim//2]

        total_pars_embedding_dim = sum(pars_embedding_dim)

        self.pars_embedding_layers = nn.ModuleList()
        for i in range(len(self.pars_dim)):
            self.pars_embedding_layers.append(
                nn.Embedding(
                    num_embeddings=pars_dims[i],
                    embedding_dim=pars_embedding_dim[i]
                )
            )
        
        self.pars_input_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.pars_input_attention = EncoderLayer(
            input_dim=total_pars_embedding_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.pars_input_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True
        )
        
        self.input_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.input_attention = DecoderLayer(
            input_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.input_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True
        )

        self.dim_reduction_layers = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=hidden_neurons[0],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[0],
                out_features=hidden_neurons[1],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[1],
                out_features=state_dim,
                bias=True
            ),
            #nn.LeakyReLU(),
        )

        self.pars_dim_reduction_layer = nn.Linear(
            in_features=total_pars_embedding_dim,
            out_features=state_dim,
            bias=True
        )

        self.pars_output_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.pars_output_attention = EncoderLayer(
            input_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.output_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.output_attention = DecoderLayer(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.output_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=False
        )


        self.pars_input_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        self.pars_output_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )


        self.input_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        self.output_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        
        '''
        self.pars_dim_increase_layer_1 = nn.Conv1d(
            in_channels=total_pars_embedding_dim,
            out_channels=latent_dim,
            kernel_size=1
        )
        self.pars_embed_increasing_layer = nn.Linear(
            in_features=1,
            out_features=embed_dim
        )

        self.pars_dim_increase_layer_2 = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=hidden_neurons[0],
            kernel_size=1
        )

        self.pars_dim_increase_layer_3 = nn.Conv1d(
            in_channels=hidden_neurons[0],
            out_channels=hidden_neurons[1],
            kernel_size=1
        )
            
        self.decoder_block1 = CrossAttentionDecoderBlock(
            input_dim=latent_dim,
            output_dim=hidden_neurons[0],
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.decoder_block2 = CrossAttentionDecoderBlock(
            input_dim=hidden_neurons[0],
            output_dim=hidden_neurons[1],
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.decoder_block3 = CrossAttentionDecoderBlock(
            input_dim=hidden_neurons[1],
            output_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        '''

    def forward(self, x, pars):

        pars = torch.cat(
            [self.pars_embedding_layers[i](pars[:, i]) for i in range(len(self.pars_dim))],
            dim=1
        )
        pars = pars.unsqueeze(-1)
        pars = self.pars_input_increase_embedding(pars)
        pars = self.activation(pars)
        pars = self.pars_input_pos_encoding(pars)
        pars = self.pars_input_attention(pars)

        x = x.unsqueeze(-1)
        x = self.input_increase_embedding(x)
        x = self.activation(x)
        x = self.input_pos_encoding(x)
        x = self.input_attention(x, pars)
        x = self.input_decrease_embedding(x)
        x = self.activation(x)
        x = x.squeeze(-1)

        pars = self.pars_input_decrease_embedding(pars)
        pars = pars.squeeze(-1)

        x = self.dim_reduction_layers(x)
        pars = self.pars_dim_reduction_layer(pars)
        pars = self.activation(pars)

        pars = pars.unsqueeze(-1)
        pars = self.pars_output_embedding(pars)
        pars = self.activation(pars)
        pars = self.pars_output_pos_encoding(pars)
        pars = self.pars_output_attention(pars)

        x = x.unsqueeze(-1)
        x = self.output_increase_embedding(x)
        x = self.activation(x)
        x = self.output_pos_encoding(x)
        x = self.output_attention(x, pars)
        x = self.output_decrease_embedding(x)
        x = self.activation(x)
        x = x.squeeze(-1)

        '''
        pars = pars.unsqueeze(-1)
        pars = self.pars_embed_increasing_layer(pars)
        pars = self.pars_dim_increase_layer_1(pars)

        x, pars = self.decoder_block1(x, pars)

        pars = self.pars_dim_increase_layer_2(pars)
        x, pars = self.decoder_block2(x, pars)

        pars = self.pars_dim_increase_layer_3(pars)
        x, _ = self.decoder_block3(x, pars)
        '''

        return x



class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int=32,
            state_dim: int=128,
            embed_dim: int=8,
            num_heads: int=2,
            hidden_neurons: list=[16, 32],
    ) -> None:
        super().__init__()


        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.activation = nn.LeakyReLU()

        self.input_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.input_attention = EncoderLayer(
            input_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.input_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True
        )

        self.dim_reduction_layers = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=hidden_neurons[0],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[0],
                out_features=hidden_neurons[1],
                bias=True
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_neurons[1],
                out_features=state_dim,
                bias=True
            ),
            nn.LeakyReLU(),
        )

        self.output_increase_embedding = nn.Linear(
            in_features=1,
            out_features=embed_dim,
            bias=True
        )
        self.output_attention = EncoderLayer(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.output_decrease_embedding = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=False
        )


        self.input_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        self.output_pos_encoding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=1000
        )
        
        '''
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(
                input_dim=latent_dim,
                output_dim=hidden_neurons[0],
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            DecoderBlock(
                input_dim=hidden_neurons[0],
                output_dim=hidden_neurons[1],
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            DecoderBlock(
                input_dim=hidden_neurons[1],
                output_dim=state_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        )
        '''

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_increase_embedding(x)
        x = self.activation(x)
        #x = self.input_pos_encoding(x)
        x = self.input_attention(x)
        x = self.input_decrease_embedding(x)
        x = self.activation(x)
        x = x.squeeze(-1)
        
        x = self.dim_reduction_layers(x)
        
        x = x.unsqueeze(-1)
        x = self.output_increase_embedding(x)
        x = self.activation(x)
        #x = self.output_pos_encoding(x)
        x = self.output_attention(x)
        x = self.output_decrease_embedding(x)
        x = x.squeeze(-1)

        #x = self.decoder_blocks(x)

        return x
