import einops
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int

# Always include batch dimension

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: Int, h: Int, dropout: float = 0.1) -> None:
        # d_model: dimension of each x
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_head = d_model // h
        self.linears = nn.Linear(d_model, 3*d_model) # output is (Q, K, V)
        self.O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.cache = None
    
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        # mask: (length, length)
        QKVOx = self.linears(x) # (b, l, 3*d)

        Qx, Kx, Vx = torch.unbind(einops.rearrange(QKVOx, "b l (three h d) -> three b l h d", h=self.h, three=3))
        attn = einops.einsum(Qx, Kx, "b lq h d, b lk h d -> b h lq lk") / self.d_head ** (1/2)
        if not mask is None:
            attn.masked_fill_(mask == 0, float('-inf'))
        softmax = torch.softmax(attn, dim=-1) # (b l l)
        post_dropout = self.dropout(softmax)

        Vx_new = einops.rearrange(einops.einsum(post_dropout, Vx, "b h lq lk,  b lk h d -> b lq h d"), "b lq h d -> b lq (h d)")

        W_Q_heads = einops.rearrange(self.linears.weight[:self.d_model], "(h d_head) d_model -> h d_head d_model", h=self.h)
        W_K_heads = einops.rearrange(self.linears.weight[self.d_model:2*self.d_model], "(h d_head) d_model -> h d_head d_model", h=self.h)
        W_QK = einops.einsum(W_Q_heads, W_K_heads, "h d_head d_q, h d_head d_k -> h d_q d_k")
        
        W_V_heads = einops.rearrange(self.linears.weight[2*self.d_model:], "(h d_head) d_model -> h d_head d_model", h=self.h)
        W_O = einops.rearrange(self.O.weight, "d_model (h d_head) -> h d_model d_head", h=self.h)
        W_OV = einops.einsum(W_O, W_V_heads, "h d_o d_head, h d_head d_v -> h d_o d_v")

        self.cache = {
            'W_QK': W_QK.detach().cpu(), # (h, d_model, d_model)
            'W_OV': W_OV.detach().cpu(), # (h, d_model, d_model)
            'attn': softmax.detach().cpu() # (b, h, l, l)
        }
        return self.O(Vx_new)
    
class MLP(nn.Module):
    def __init__(self, d_model: Int, d_ffn: Int) -> None:
        super().__init__()
        assert d_ffn % d_model == 0
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.ffn_1 = nn.Linear(d_model, d_ffn)
        self.ffn_2 = nn.Linear(d_ffn, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        return self.ffn_2(self.relu(self.ffn_1(x)))
    
class LayerNorm(nn.Module):
    def __init__(self, shape: Int, eps: Float = 1e-6) -> None:
        # shape = d_model
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape))
        self.b = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x: Float[Tensor, "batch length d_model"]) -> Float[Tensor, "batch length d_model"]:
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return ((x - mean) / (std + self.eps)) * self.a + self.b
    
class SubLayer(nn.Module):
    def __init__(self, shape: Int, layer_fn: nn.Module, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.layer_fn = layer_fn  # Either Attention or MLP
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNorm(shape)
        else:
            self.layer_norm = None
    
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        if self.use_layer_norm:
            return self.layer_norm(x + self.layer_fn(x, mask))
        else:
            return x + self.layer_fn(x, mask)

import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model: Int, d_ffn: Int, h: Int, attn_only: bool = False, layer_norm: bool = True) -> None:
        super().__init__()
        self.attn_only = attn_only
        self.layer_norm = layer_norm
        self.attn = SubLayer(d_model, MultiHeadAttention(d_model, h), self.layer_norm)
        if self.attn_only:
            self.mlp = None
        else:
            self.mlp = SubLayer(d_model, MLP(d_model, d_ffn), self.layer_norm)
        
    
    def forward(self, x: Float[Tensor, "batch length d_model"], mask: Optional[Tensor] = None) -> Float[Tensor, "batch length d_model"]:
        if self.attn_only:
            return self.attn(x, mask)
        else:
            return self.mlp(self.attn(x, mask))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: Int, max_len: Int = 2048) -> None:
        super(PositionalEncoding, self).__init__()
        
        weight = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
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
    def __init__(self, vocab_size: Int = 128, d_model: Int = 768, d_ffn: Int = 3072, h: Int = 12, n: Int = 2, max_len: Int = 2048, attn_only: bool = False, layer_norm: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.h = h
        self.n = n
        self.attn_only = attn_only
        self.layer_norm = layer_norm
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ffn, h, attn_only, layer_norm) for _ in range(self.n)])
        self.pos = PositionalEncoding(d_model, max_len)
        self.W_U = nn.Linear(d_model, vocab_size)
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