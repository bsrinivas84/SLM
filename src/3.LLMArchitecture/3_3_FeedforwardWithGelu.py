"""Implement GELU and a transformer feed-forward network."""

#Gelu adding Non-linearity for learning complex representations
#  and Feedforward layer for transformer block

# Transformer block flow (pre-norm with shortcut connections):
#
# x
#  -> LayerNorm 1
#  -> Masked Multi-Head Attention
#  -> Dropout
#  -> + residual(x)
#  -> LayerNorm 2
#  -> FeedForward [Linear -> GELU -> Linear]
#  -> Dropout
#  -> + residual(from first shortcut output)
#  -> block output

import torch
import torch.nn as nn
import importlib


GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}


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


if __name__ == "__main__":
    gelu, relu = GELU(), nn.ReLU()

    # Some sample data.
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt = None
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping GELU/ReLU plot.")

    if plt is not None:
        plt.figure(figsize=(8, 3))
        for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
            plt.subplot(1, 2, i)
            plt.plot(x, y)
            plt.title(f"{label} activation function")
            plt.xlabel("x")
            plt.ylabel(f"{label}(x)")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    GP = {"emb_dim": 768}
    ffn = FeedForward(GP)
    sample = torch.randn(2, 3, GP["emb_dim"])
    out = ffn(sample)
    print("FeedForward output shape:", out.shape)




