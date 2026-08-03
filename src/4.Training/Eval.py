from PreviousChapters import GPTModel, generate_text_simple
import torch
import tiktoken


GPT_CONFIG_124 = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(cfg=GPT_CONFIG_124)
model.eval()  # Set the model to evaluation mode


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension.
    return encoded_tensor


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = text_to_token_ids(start_context, tokenizer)
print(token_ids)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Remove batch dimension.
    return tokenizer.decode(flat.tolist())

tokens = token_ids_to_text(token_ids, tokenizer)
print(tokens)


token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124["context_length"],
)

print(token_ids.squeeze(0).shape)
print(token_ids_to_text(token_ids, tokenizer))


