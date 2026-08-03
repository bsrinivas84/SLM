"""Calculate language-model losses over data loaders."""

import torch
from torch import nn
import tiktoken
from torch.return_types import mode
tokenizer = tiktoken.get_encoding("gpt2")
import PreviousChapters

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

with open('../../data/raw/Verdict.txt', 'r',encoding = "utf-8") as file:
    text_data = file.read()
    text_data_length = len(text_data)
    print(text_data_length) #20479


print(text_data[:100]) #
print("Total Tokens:", len(tokenizer.encode(text_data))) # Print the total number of tokens in the text data

#Subset of 
training = text_data[:10000]
validation = text_data[10000:len(text_data)]

train_ratio = 0.9
split_index = int(text_data_length * train_ratio)

training_data = text_data[:split_index]
validation_data = text_data[split_index:text_data_length]
print(f"Training data length: {len(training_data)}")
print(f"Validation data length: {len(validation_data)}")

train_loader = PreviousChapters.create_dataloader_v1(
    training_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True, 
    shuffle=True,
    num_workers=0
)
val_loader = PreviousChapters.create_dataloader_v1(
    validation_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True, 
    shuffle=True,
    num_workers=0
)

print("Train Loader")
for x,y in train_loader:
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    break
print("Validation Loader")
for x,y in val_loader:
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    break   

train_tokens = 0 

for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()  # Count the number of tokens in the input batch
print("Total training tokens:", train_tokens)

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()  # Count the number of tokens in the input batch
print("Total validation tokens:", val_tokens)   
print("Total tokens (train + validation):", train_tokens + val_tokens)


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Return scalar cross-entropy loss for ``(batch, tokens)`` input and target IDs."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, num_batches: int | None = None) -> float:
    """Return mean batch loss, or NaN when the loader is empty."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader.
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PreviousChapters.GPTModel(cfg=GPT_CONFIG_124M).to(device)

print("Training loss:", calc_loss_loader(train_loader, model, device))
print("Validation loss:", calc_loss_loader(val_loader, model, device))
