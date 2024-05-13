import einops
import math
import torch
import torch.nn as nn

# Always include batch dimension

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        # d_model: dimension of each x
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_head = d_model // h
        self.linears = nn.Linear(d_model, 3*d_model) # output is (Q, K, V)
        self.O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        # x: (batch, length, d_model)
        # mask: (length, length)
        QKVO = self.linears(x) # (b, l, 3*d)
        Q, K, V = torch.unbind(einops.rearrange(QKVO, "b l (three h d) -> three b l h d", h=self.h, three=3))
        QK = einops.einsum(Q, K, "b lq h d, b lk h d -> b h lq lk") / self.d_head ** (1/2)
        if not mask is None:
            QK.masked_fill_(mask == 0, float('-inf'))
        softmax = torch.softmax(QK, dim=-1) # (b l l), is it divide by d_head?

        attn = einops.rearrange(einops.einsum(softmax, V, "b h lq lk,  b lk h d -> b lq h d"), "b lq h d -> b lq (h d)")
        return self.O(attn)
    
class MLP(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        assert d_ffn % d_model == 0
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.ffn_1 = nn.Linear(d_model, d_ffn)
        self.ffn_2 = nn.Linear(d_ffn, d_model)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        return self.ffn_2(self.relu(self.ffn_1(x)))
    
class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        # shape = d_model
        super().__init__()
        self.a = nn.Parameter(torch.ones(shape))
        self.b = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        # x: (batch, length, d_model)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return ((x - mean) / (std + self.eps)) * self.a + self.b
    
class SubLayer(nn.Module):
    def __init__(self, shape, layer_fn):
        super().__init__()
        self.layer_fn = layer_fn # Either Attention or MLP
        self.layer_norm = LayerNorm(shape)
    
    def forward(self, x, mask=None):
        # x: (batch, length, dimension)
        return self.layer_norm(x + self.layer_fn(x, mask))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, h):
        super().__init__()
        self.attn = SubLayer(d_model, MultiHeadAttention(d_model, h))
        self.mlp = SubLayer(d_model, MLP(d_model, d_ffn))
    
    def forward(self, x, mask=None):
        return self.mlp(self.attn(x, mask))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=96, d_ffn=3072, h=12, n=2, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.h = h
        self.n = n
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ffn, h) for _ in range(self.n)])
        self.pe = PositionalEncoding(d_model, max_len)
        self.W_O = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            print(p.shape)

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, length, d_model)
        x = self.pe(self.W_E(x))
        mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).type(torch.uint8) == 0
        for layer in self.layers:
            x = layer(x, mask)
            print(x)

        return self.W_O(x[:, -1, :]).softmax(dim=-1)
    
transformer = Transformer()
input = einops.rearrange(torch.arange(32), "(b l) -> b l", b = 4)
print(input)

for _ in range(8):
    output = torch.argmax(transformer(input), dim=-1)
    print(output)
    input = torch.cat([input, output.unsqueeze(-1)], dim=-1)

print(input)
