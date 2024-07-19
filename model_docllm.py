import math

import inspect
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.nn import functional as F


class DocLLMConfig(BaseModel):
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    bias: bool = True
    dropout: float = 0.0
    spatial_precision: int


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, n_embd: int, bias: bool):
        """
        Paremeters
        ----------
        n_embd: int
            Number of features in the input tensor
        bias: bool
            If True, adds a learnable bias to the output tensor
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """(B, ..., ndim) -> (B, ..., ndim)"""
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class DisentangledSpatialAttention(nn.Module):

    def __init__(self, config: DocLLMConfig):
        """
        Parameters
        ----------
        config: DocLLMConfig
            All parameters of the model
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads of token embeddings
        self.t_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # key, query projections for all heads of spatial embeddings
        self.s_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.lambda_ts = nn.Parameter(torch.ones(1))
        self.lambda_st = nn.Parameter(torch.ones(1))
        self.lambda_ss = nn.Parameter(torch.ones(1))

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, xt: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, n_embd), (B, seq_len, n_embd) -> (B, seq_len, n_embd)"""
        B, seq_len, n_embd = xt.size()

        # calculate query, key, values for all heads
        qt: torch.Tensor
        kt: torch.Tensor
        vt: torch.Tensor
        qs: torch.Tensor
        ks: torch.Tensor

        # (B, seq_len, 3 * n_embd) -> (B, seq_len, n_embd) x 3
        qt, kt, vt = self.t_attn(xt).split(self.n_embd, dim=2)
        # (B, seq_len, 2 * n_embd) -> (B, seq_len, n_embd) x 2
        qs, ks = self.s_attn(xs).split(self.n_embd, dim=2)

        # h_embd = n_embd // n_head
        # (B, seq_len, n_embd) -> (B, n_head, seq_len, h_embd)
        kt = kt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        qt = qt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        vt = vt.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)

        ks = ks.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)
        qs = qs.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)

        # attention
        att: torch.Tensor
        y: torch.Tensor
        att = (
            (qt @ kt.transpose(-2, -1))
            + self.lambda_ts * (qt @ ks.transpose(-2, -1))
            + self.lambda_st * (qs @ kt.transpose(-2, -1))
            + self.lambda_ss * (qs @ ks.transpose(-2, -1))
        ) * (1.0 / math.sqrt(kt.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ vt

        # (B, n_head, seq_len, h_embd) -> (B, seq_len, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, seq_len, n_embd)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, config: DocLLMConfig):
        """
        Parameters
        ----------
        config: DocLLMConfig
            All parameters of the model
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, n_embd) -> (B, seq_len, n_embd)"""
        xt = self.c_fc(xt)
        xt = self.gelu(xt)
        xt = self.c_proj(xt)
        xt = self.dropout(xt)
        return xt


class Block(nn.Module):

    def __init__(self, config: DocLLMConfig):
        """
        Parameters
        ----------
        config: DocLLMConfig
            All parameters of the model
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = DisentangledSpatialAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, xt: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, n_embd), (B, seq_len, n_embd) -> (B, seq_len, n_embd)"""
        xt = xt + self.attn(self.ln_1(xt), xs)
        xt = xt + self.mlp(self.ln_2(xt))
        return xt


class DocLLM(nn.Module):
    def __init__(self, config: DocLLMConfig):
        """
        Parameters
        ----------
        config: DocLLMConfig
            All parameters of the model
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                spe=nn.Linear(4 * config.spatial_precision, config.n_embd),
                drop_t=nn.Dropout(config.dropout),
                drop_s=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, bbox, targets=None):
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), "Cannot forward sequence of length {}, block size is only {}".format(
            t, self.config.block_size
        )

        xt = self.transformer.wte(idx)  # (b, t, n_embd)
        xs = self.transformer.spe(bbox)  # (b, t, n_embd)

        xt = self.transformer.drop_t(xt)
        xs = self.transformer.drop_s(xs)

        for block in self.transformer.h:
            xt = block(xt, xs)
        xt = self.transformer.ln_f(xt)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(xt)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                xt[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, bbox, max_new_tokens, B_MASK, temperature=1.0, top_k=None):
        """
        idx (b, t)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it from the end
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            bbox_cond = (
                bbox
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, _ = self(idx_cond, bbox_cond)
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Bounding box that just got filled
            bbox_next = B_MASK
            idx = torch.cat((idx, idx_next), dim=1)
            bbox = torch.cat((bbox, bbox_next), dim=1)

        return idx
