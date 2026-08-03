"""Implement causal attention with dropout."""

import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """Compute batched causal self-attention with dropout."""
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Create a causal mask to prevent attention to future tokens
        #As its a random tensor, register the mask as a buffer so it moves with the model to the appropriate device
        self.register_buffer("mask_var", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``(batch, tokens, d_in)`` inputs to ``(batch, tokens, d_out)``."""
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        b, num_tokens, d_in = x.shape

        attn_scores = queries @ keys.transpose(1, 2)
    # Set future-token scores to -inf so softmax turns them into 0, enforcing causal (past-only) attention.
        attn_scores.masked_fill_(self.mask_var.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        #Drop out masked attention weights to prevent overfitting and improve generalization
        attn_weights = self.dropout(attn_weights)

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

batch = torch.stack([inputs, inputs])  # Create a batch of size 2

torch.manual_seed(789)
d_in = inputs.shape[1]
d_out = 2
dropout = 0.0
ca = CausalAttention(d_in, d_out, context_length=inputs.shape[0], dropout=0.0)
print(ca(batch))
