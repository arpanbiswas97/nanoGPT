import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pydantic import BaseModel


class DocLLMConfig(BaseModel):
    n_embd: int
    n_head: int
    bias: float
    dropout: float


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2, ndim)))
        self.bias = nn.Parameter(torch.zeros((2, ndim))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class DisentangledSpatialAttention(nn.Module):

    def __init__(self, config: DocLLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.t_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.s_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.lambda_ts = nn.Parameter(torch.ones(1))
        self.lambda_st = nn.Parameter(torch.ones(1))
        self.lambda_ss = nn.Parameter(torch.ones(1))

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, _, n_embd = x.size()

        # (B, seq_len, 2, n_embd) -> (B, seq_len, n_embd), (B, seq_len, n_embd)
        xt, xs = torch.split(x, 1, dim=2)
        xt, xs = xt.squeeze(), xs.squeeze()

        # calculate query, key, values for all heads
        qt, kt, vt = self.t_attn(xt).split(self.n_embd, dim=2)
        qs, ks = self.s_attn(xs).split(self.n_embd, dim=2)

        # (B, seq_len, n_embd) -> (B, nh, seq_len, hs)
        kt = kt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        qt = qt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        vt = vt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)

        ks = ks.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        qs = qs.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)

        # attention
        att = (
            (qt @ kt.transpose(-2, -1))
            + self.lambda_ts * (qt @ ks.transpose(-2, -1))
            + self.lambda_st * (qs @ kt.transpose(-2, -1))
            + self.lambda_ss * (qs @ ks.transpose(-2, -1))
        ) * (1.0 / math.sqrt(kt.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ vt

        # (B, nh, seq_len, hs) -> (B, seq_len, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, seq_len, n_embd)
        y = self.resid_dropout(self.out_proj(y))

        # (B, seq_len, 2, n_embd)
        yns = torch.cat([y.unsqueeze(2), xs.unsqueeze(2)], dim=2).contiguous()

        return yns


class MLP(nn.Module):

    def __init__(self, config: DocLLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xt, xs = torch.split(x, 1, dim=2)
        xt = xt.squeeze()

        xt = self.c_fc(xt)
        xt = self.gelu(xt)
        xt = self.c_proj(xt)
        xt = self.dropout(xt)

        x = torch.cat([xt.unsqueeze(2), xs], dim=2).contiguous()
        return x


class DocLLM(nn.Module):
    ...