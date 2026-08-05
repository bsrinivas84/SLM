"""Prepare tokenized instruction data for batching."""

from typing import Any, Protocol

import torch
from torch.utils.data import Dataset


class Tokenizer(Protocol):
	def encode(self, text: str) -> list[int]: ...


def format_input(entry: dict[str, Any]) -> str:
	"""Format an instruction dataset entry as a model prompt."""
	instruction_text = (
		"Below is an instruction that describes a task. "
		"Write a response that appropriately completes the request."
		f"\n\n### Instruction:\n{entry['instruction']}"
	)
	input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

	return instruction_text + input_text


class InstructionDataset(Dataset):
	"""Pre-tokenize instruction-response examples for a data loader."""

	def __init__(self, data: list[dict[str, Any]], tokenizer: Tokenizer) -> None:
		self.data = data
		self.encoded_texts = []

		for entry in data:
			instruction_plus_input = format_input(entry)
			response_text = f"\n\n### Response:\n{entry['output']}"
			full_text = instruction_plus_input + response_text
			self.encoded_texts.append(tokenizer.encode(full_text))

	def __getitem__(self, index: int) -> list[int]:
		return self.encoded_texts[index]

	def __len__(self) -> int:
		return len(self.data)


def custom_collate(
	batch: tuple[list[int], ...] | list[list[int]],
	pad_token_id: int = 50256,
	ignore_index: int = -100,
	allowed_max_length: int | None = None,
	device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Pad token sequences and create next-token prediction targets."""
	if not batch:
		raise ValueError("batch must contain at least one sequence")

	batch_max_length = max(len(item) + 1 for item in batch)
	inputs_list, targets_list = [], []

	for item in batch:
		new_item = item.copy()
		new_item.append(pad_token_id)
		padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

		inputs = torch.tensor(padded[:-1])
		targets = torch.tensor(padded[1:])

		mask = targets == pad_token_id
		indices = torch.nonzero(mask).squeeze(-1)
		if indices.numel() > 1:
			targets[indices[1:]] = ignore_index

		if allowed_max_length is not None:
			inputs = inputs[:allowed_max_length]
			targets = targets[:allowed_max_length]

		inputs_list.append(inputs)
		targets_list.append(targets)

	inputs_tensor = torch.stack(inputs_list).to(device)
	targets_tensor = torch.stack(targets_list).to(device)
	return inputs_tensor, targets_tensor


def compare_cross_entropy_losses() -> tuple[torch.Tensor, torch.Tensor]:
	"""Compare loss with a regular target and an ignored target."""
	logits = torch.tensor(
		[
			[-1.0, 1.0],
			[-0.5, 1.5],
			[-0.6, 1.6],
		]
	)

	loss_without_mask = torch.nn.functional.cross_entropy(
		logits, torch.tensor([0, 1, 1])
	)
	loss_with_mask = torch.nn.functional.cross_entropy(
		logits, torch.tensor([0, 1, -100])
	)

	return loss_without_mask, loss_with_mask


if __name__ == "__main__":
	loss_without_mask, loss_with_mask = compare_cross_entropy_losses()
	print(loss_without_mask)
	print(loss_with_mask)
