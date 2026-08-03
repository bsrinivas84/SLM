import torch

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_default_device()
print(f"Using device: {device}")

#input_ids = torch.tensor([1, 2, 3, 4, 5], device=device)

torch.manual_seed(123)  # For reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

vocab_size = 50257
output_dim = 256
embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)  # 3-dimensional embeddings
print(embedding_layer.weight)  # Print the initial weights