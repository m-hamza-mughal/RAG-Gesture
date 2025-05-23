import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float()
        div_term = div_term * (-np.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # T, 1, D
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


# ConvoFusion repo:

class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model, max_len=1024, batch_first=False): 
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            # breakpoint()
            pos = x + self.pe[:x.shape[0], :]
        return pos

class PositionEmbeddingSineBH(nn.Module):

    def __init__(self, d_model, max_len=1024, batch_first=False): 
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            # breakpoint()
            x[0::2] = x[0::2] + self.pe[:x.shape[0]//2, :]
            x[1::2] = x[1::2] + self.pe[:x.shape[0]//2, :]
            pos = x
        return pos


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model, max_len=1024, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        # self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x
        # return self.dropout(x)


def build_position_encoding(N_steps,
                            position_embedding="sine",
                            embedding_dim="1D"):
    # N_steps = hidden_dim // 2
    
    if embedding_dim == "1D":
        if position_embedding in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine1D(N_steps)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned1D(N_steps)
        elif position_embedding in ('v2', 'sine_bh'):
            position_embedding = PositionEmbeddingSineBH(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    elif embedding_dim == "2D":
        if position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    else:
        raise ValueError(f"not supported {embedding_dim}")

    return position_embedding