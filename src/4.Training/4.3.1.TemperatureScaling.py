"""Demonstrate temperature-based token sampling."""

import matplotlib.pyplot as plt
import tiktoken
import torch

import PreviousChapters


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Encode text as a token tensor shaped ``(1, tokens)``."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Decode a token-ID tensor into text."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """Append generated IDs to a ``(batch, tokens)`` tensor and return the result."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def print_sampled_tokens(probas: torch.Tensor, inverse_vocab: dict[int, str]) -> None:
    """Print sampled frequency counts for a one-dimensional probability tensor."""
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(inverse_vocab))
    for index, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[index]}")


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return temperature-scaled probabilities with the same shape as ``logits``."""
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


if __name__ == "__main__":
    torch.manual_seed(123)
    model = PreviousChapters.GPTModel(cfg=GPT_CONFIG_124M)
    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {value: key for key, value in vocab.items()}
    print(inverse_vocab)

    # model(x)
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    probas = torch.softmax(next_token_logits, dim=0)
    print(probas)

    next_token_id = torch.argmax(probas).item()
    print(next_token_id)
    print(inverse_vocab[next_token_id])

    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    print_sampled_tokens(probas, inverse_vocab)

    temperatures = [1, 0.1, 5]

    # Calculate scaled probabilities.
    scaled_probas = [softmax_with_temperature(next_token_logits, temperature) for temperature in temperatures]

    print(scaled_probas[1])
    print(scaled_probas[2])

    # Plotting.
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for index, temperature in enumerate(temperatures):
        ax.bar(x + index * bar_width, scaled_probas[index], bar_width, label=f"Temperature = {temperature}")

    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig("temperature-plot.pdf")
    plt.show()
