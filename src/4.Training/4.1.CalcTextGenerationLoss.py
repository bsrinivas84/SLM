from PreviousChapters import GPTModel
import torch
import tiktoken


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())


torch.manual_seed(123)
model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])

targets = torch.tensor([[3626, 6100, 345],
                        [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)

print("logits.shape:", logits.shape)

probas = torch.softmax(logits, dim=-1)
print("probas.shape:", probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# Compute logarithm of all token probabilities.
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_neg_log_proba = -1 * torch.mean(log_probas)
print("Average Negative Log Probability:", avg_neg_log_proba)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

ce_loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("cross_entropy:", ce_loss)

loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
print("Cross Entropy Loss:", loss)