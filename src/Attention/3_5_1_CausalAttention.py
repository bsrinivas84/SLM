# Mask the future words
import torch.nn as nn
import torch


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=bias)
        self.W_values = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        attn_scores = queries @ keys.T
        print("Attention scores:\n", attn_scores)
        mask =  torch.tril(torch.ones_like(attn_scores))  # Mask future words
        attn_scores_masked = attn_scores.masked_fill(mask == 0, float('-inf'))  # Apply the mask
        print("Mask:\n", mask)
        print("Masked attention scores:\n", attn_scores_masked)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores_masked / (d_k ** 0.5), dim=-1)
        print("Attention weights:\n", attn_weights)
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
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))