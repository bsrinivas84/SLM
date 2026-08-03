"""Implement compact self-attention modules."""

import torch
import torch.nn as nn




class SelfAttention_v1(nn.Module):
    """Compute scaled dot-product self-attention with parameter matrices."""
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``(tokens, d_in)`` inputs into ``(tokens, d_out)`` context vectors."""
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        context_vec = attn_weights @ values

        return context_vec

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
torch.manual_seed(123)
d_in = inputs.shape[1]
d_out = 2
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

#======================================================


class SelfAttention_v2(nn.Module):
    """Compute scaled dot-product self-attention with linear projections."""
    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=bias)
        self.W_values = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``(tokens, d_in)`` inputs into ``(tokens, d_out)`` context vectors."""
        queries = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        context_vec = attn_weights @ values

        return context_vec

sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
