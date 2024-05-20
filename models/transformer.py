import einops
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 128
    d_model: int = 768
    d_ffn: int = 3072
    h: int = 12
    d_head: int = d_model // h
    n: int = 2
    max_len: int = 2048
    attn_only: bool = False
    layer_norm: bool = True
    ln_eps: float = 1e-6
    dropout: float = 0.1
# Always include batch dimension

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        # d_model: dimension of each x
        super().__init__()
        assert config.d_model % config.h == 0
        self.config = config
        self.linears = nn.Linear(config.d_model, 3*config.d_model) # output is (Q, K, V)
        self.O = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.cache = None
    
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        # mask: (length, length)
        QKVOx = self.linears(x) # (b, l, 3*d)

        Qx, Kx, Vx = torch.unbind(einops.rearrange(QKVOx, "b l (three h d) -> three b l h d", h=self.config.h, three=3))
        attn = einops.einsum(Qx, Kx, "b lq h d, b lk h d -> b h lq lk") / self.config.d_head ** (1/2)
        if not mask is None:
            attn.masked_fill_(mask == 0, float('-inf'))
        softmax = torch.softmax(attn, dim=-1) # (b l l)
        post_dropout = self.dropout(softmax)

        Vx_new = einops.rearrange(einops.einsum(post_dropout, Vx, "b h lq lk,  b lk h d -> b lq h d"), "b lq h d -> b lq (h d)")

        W_Q_heads = einops.rearrange(self.linears.weight[:self.config.d_model], "(h d_head) d_model -> h d_head d_model", h=self.config.h)
        W_K_heads = einops.rearrange(self.linears.weight[self.config.d_model:2*self.config.d_model], "(h d_head) d_model -> h d_head d_model", h=self.config.h)
        W_QK = einops.einsum(W_Q_heads, W_K_heads, "h d_head d_q, h d_head d_k -> h d_q d_k")
        
        W_V_heads = einops.rearrange(self.linears.weight[2*self.config.d_model:], "(h d_head) d_model -> h d_head d_model", h=self.config.h)
        W_O = einops.rearrange(self.O.weight, "d_model (h d_head) -> h d_model d_head", h=self.config.h)
        W_OV = einops.einsum(W_O, W_V_heads, "h d_o d_head, h d_head d_v -> h d_o d_v")

        self.cache = {
            'W_QK': W_QK.detach().cpu(), # (h, d_model, d_model)
            'W_OV': W_OV.detach().cpu(), # (h, d_model, d_model)
            'attn': softmax.detach().cpu() # (b, h, l, l)
        }
        return self.O(Vx_new)
    
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.d_ffn % config.d_model == 0
        self.ffn_1 = nn.Linear(config.d_model, config.d_ffn)
        self.ffn_2 = nn.Linear(config.d_ffn, config.d_model)
        self.relu = nn.ReLU()

    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        return self.ffn_2(self.relu(self.ffn_1(x)))
    
class LayerNorm(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        # shape = d_model
        super().__init__()
        self.config = config
        self.a = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, x: Float[Tensor, "batch length d_model"]) -> Float[Tensor, "batch length d_model"]:
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return ((x - mean) / (std + self.config.ln_eps)) * self.a + self.b
    
class SubLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_fn: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.layer_fn = layer_fn  # Either Attention or MLP
        if config.layer_norm:
            self.layer_norm = LayerNorm(config)
        else:
            self.layer_norm = None
    
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        if self.config.layer_norm:
            return self.layer_norm(x + self.layer_fn(x, mask))
        else:
            return x + self.layer_fn(x, mask)

class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = SubLayer(config, MultiHeadAttention(config))
        if config.attn_only:
            self.mlp = None
        else:
            self.mlp = SubLayer(config, MLP(config))
        
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        if self.config.attn_only:
            return self.attn(x, mask)
        else:
            return self.mlp(self.attn(x, mask))
    
class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(PositionalEncoding, self).__init__()
        self.config = config
        weight = torch.zeros(config.max_len, config.d_model)
        position = torch.arange(0, config.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0)
        self.register_buffer("weight", weight)

    def forward(self, x: Float[Tensor, "batch length d_model"]) -> Float[Tensor, "batch length d_model"]:
        x = x + self.weight[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    """
    Defines a transformer model with causal attention. 
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.W_E = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n)])
        self.pos = PositionalEncoding(config)
        self.W_U = nn.Linear(config.d_model, config.vocab_size)
        self.cache = {
            "mask": None,
            "W_E": self.W_E.weight.detach().cpu(),
            "W_U": self.W_U.weight.detach().cpu(),
            "W_pos": self.pos.weight[0].detach().cpu()
        }
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Float[Tensor, "batch length d_model"]) -> Float[Tensor, "batch length d_model"]:
        x = self.pos(self.W_E(x))
        mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).type(torch.float32) == 0
        mask = mask.to(x.device)
        self.cache["mask"] = mask.detach().cpu()

        for layer in self.layers:
            x = layer(x, mask)

        return self.W_U(x)