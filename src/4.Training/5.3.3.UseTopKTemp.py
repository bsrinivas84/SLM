import torch
import tiktoken

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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.0,
                         top_k=None, eos_token_id=None):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) 

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        new_logits = logits.clone()
        # Apply Top-K sampling if specified
        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            new_logits = torch.where(
                condition=logits < top_logits[:, -1, None],
                input=torch.tensor(float("-inf")),
                other=logits,
            )
        
        #Apply temperature scaling if specified
        if temperature > 0.0:
            new_logits = new_logits / temperature
            probs = torch.softmax(new_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
        else:
            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(new_logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        if eos_token_id is not None and idx_next == eos_token_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":
    model = PreviousChapters.GPTModel(GPT_CONFIG_124M)
    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))