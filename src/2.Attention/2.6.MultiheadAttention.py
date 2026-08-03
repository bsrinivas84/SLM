import torch
import torch.nn as nn
from pathlib import Path
import importlib.util


dropout_mask_path = Path(__file__).with_name("2_5_2DropoutMask.py")
spec = importlib.util.spec_from_file_location("dropout_mask_module", dropout_mask_path)
dropout_mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dropout_mask_module)
CausalAttention = dropout_mask_module.CausalAttention



class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias, num_heads=2):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for
    _ in range(num_heads)])
        

    def forward(self, x):
        # Concatenate the outputs from all heads along the last dimension
        return torch.cat([head(x) for head in self.heads], dim=-1) 
    


torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

batch = torch.stack([inputs, inputs])  # Create a batch of size 2

context_length = batch.shape[1]
d_in,d_out = 3, 2

MultiheadAttention_model = MultiheadAttention(d_in, d_out, context_length, dropout=0.0, qkv_bias=False, num_heads=2)
print("Multihead context vectors", MultiheadAttention_model(batch))

