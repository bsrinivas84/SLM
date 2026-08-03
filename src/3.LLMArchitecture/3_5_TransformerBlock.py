"""Implement a pre-normalized transformer block."""

import torch
import torch.nn as nn

from MultiHeadAttentionRef import MultiHeadAttention


class LayerNorm(nn.Module):
    """Normalize activations over their final dimension."""
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor while preserving its shape."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Apply the Gaussian error linear unit approximation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU elementwise and return a tensor with the input shape."""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Expand, activate, and project transformer representations."""
    def __init__(self, cfg: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform embeddings while preserving ``(batch, tokens, emb_dim)``."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Apply causal attention and feed-forward residual sublayers."""
    def __init__(self, cfg: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual shortcut.
        """Transform and return ``(batch, tokens, emb_dim)`` activations."""
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward block with residual shortcut.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    cfg = {
        "emb_dim": 768,
        "context_length": 1024,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    block = TransformerBlock(cfg)
    sample = torch.randn(2, 4, cfg["emb_dim"])
    out = block(sample)
    print("TransformerBlock output shape:", out.shape)

