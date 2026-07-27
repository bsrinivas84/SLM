import importlib.util
from pathlib import Path

import tiktoken
import torch


_gpt_path = Path(__file__).with_name("4_6_GPTModel.py")
_spec = importlib.util.spec_from_file_location("gpt_model_module", _gpt_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GPTModel = _module.GPTModel
GPT_CONFIG_124M = _module.GPT_CONFIG_124M


def generate_text_simple(model, idx, max_new_tokens, context_size):
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]

		with torch.no_grad():
			logits = model(idx_cond)

		logits = logits[:, -1, :]
		probas = torch.softmax(logits, dim=-1)
		idx_next = torch.argmax(probas, dim=-1, keepdim=True)
		idx = torch.cat((idx, idx_next), dim=1)

	return idx


if __name__ == "__main__":
	tokenizer = tiktoken.get_encoding("gpt2")

	start_context = "Hello, I am"
	encoded = tokenizer.encode(start_context)
	print("encoded:", encoded)

	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	print("encoded_tensor.shape:", encoded_tensor.shape)

	print(torch.argmax(torch.tensor([14, 1, -2, 1, 15])))

	torch.manual_seed(123)
	model = GPTModel(cfg=GPT_CONFIG_124M)

	out = generate_text_simple(
		model=model,
		idx=encoded_tensor,
		max_new_tokens=6,
		context_size=GPT_CONFIG_124M["context_length"],
	)

	print("generated token ids:", out)
	print("decoded:", tokenizer.decode(out[0].tolist()))
