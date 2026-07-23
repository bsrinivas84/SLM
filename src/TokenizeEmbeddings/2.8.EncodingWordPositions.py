from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


def get_default_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


class GPTDatasetV1(Dataset):
	def __init__(self, txt, tokenizer, max_length, stride):
		self.input_ids = []
		self.target_ids = []

		token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

		for i in range(0, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i:i + max_length]
			target_chunk = token_ids[i + 1:i + max_length + 1]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=2, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
		num_workers=num_workers,
	)
	return dataloader


def main():
	device = get_default_device()
	print(f"Using device: {device}")

	data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "Verdict.txt"
	raw_text = data_path.read_text(encoding="utf-8")

	max_length = 4
	dataloader = create_dataloader_v1(
		raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
	)

	data_iter = iter(dataloader)
	inputs, targets = next(data_iter)

	print("Token IDs:\n", inputs)
	print("\nInputs shape:\n", inputs.shape)
	print("Target IDs:\n", targets)


	vocab_size = 50257
	output_dim = 256
	token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)

	inputs = inputs.to(device)
	token_embeddings = token_embedding_layer(inputs)
	print("\nToken embeddings shape:\n", token_embeddings.shape)

	pos_embedding_layer = torch.nn.Embedding(max_length, output_dim).to(device) # these are random weights, actual once are learnt later
	print("\nPosition embedding weights:\n", pos_embedding_layer.weight)

	pos_embeddings = pos_embedding_layer(torch.arange(max_length, device=device))
	print("\nPosition embeddings shape:\n", pos_embeddings.shape)
	print("\nToken embeddings shape:\n", token_embeddings.shape)

	input_embeddings = token_embeddings + pos_embeddings
	print("\nInput embeddings shape:\n", input_embeddings.shape)


if __name__ == "__main__":
	main()
