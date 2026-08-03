"""Define a placeholder GPT model for architecture exploration."""

import torch
import torch.nn as nn
import tiktoken


GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}


class DummyTransformerBlock(nn.Module):
    """Pass transformer activations through unchanged."""
    def __init__(self, cfg: dict[str, int | float | bool]) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged, preserving its shape."""
        return x


class DummyLayerNorm(nn.Module):
    """Pass activations through unchanged in place of normalization."""
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged, preserving its shape."""
        return x


class DummyGPTModel(nn.Module):
    """Expose the GPT architecture using placeholder transformer operations."""
    def __init__(self, cfg: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock.
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Use a placeholder for LayerNorm.
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        #The output head is a linear layer that maps the final hidden states to the vocabulary size for token prediction.
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(batch, tokens, vocab_size)`` for token IDs."""
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(cfg=GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)


