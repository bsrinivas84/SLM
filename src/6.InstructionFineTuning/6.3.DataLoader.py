import torch
import tiktoken
from functools import partial
import importlib.util
from types import ModuleType
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


def import_prepare_dataset() -> ModuleType:
	"""Load and return the neighboring dataset-preparation module."""
	module_path = Path(__file__).with_name("6.1.PrepareDataset.py")
	spec = importlib.util.spec_from_file_location("PrepareDataset", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def import_OrganizeBatch() -> ModuleType:
	"""Load and return the neighboring loss-calculation module."""
	module_path = Path(__file__).with_name("6.2.OrganizeBatches.py")
	spec = importlib.util.spec_from_file_location("OrganizeBatch", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

OrganizeBatch = import_OrganizeBatch()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

customized_collate_fn = partial(
    OrganizeBatch.custom_collate,
    device=device,
    allowed_max_length=1024
)


def create_train_loader(
	train_data: list[dict[str, Any]],
	tokenizer: Any,
	batch_size: int = 8,
	num_workers: int = 0,
) -> DataLoader:
	"""Create a shuffled instruction-training data loader."""
	torch.manual_seed(123)
	train_dataset = OrganizeBatch.InstructionDataset(train_data, tokenizer)

	return DataLoader(
		train_dataset,
		batch_size=batch_size,
		collate_fn=customized_collate_fn,
		shuffle=True,
		drop_last=True,
		num_workers=num_workers,
	)


def create_val_loader(
	val_data: list[dict[str, Any]],
	tokenizer: Any,
	batch_size: int = 8,
	num_workers: int = 0,
) -> DataLoader:
	"""Create an ordered validation data loader that retains the final batch."""
	val_dataset = OrganizeBatch.InstructionDataset(val_data, tokenizer)

	return DataLoader(
		val_dataset,
		batch_size=batch_size,
		collate_fn=customized_collate_fn,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers,
	)


def create_test_loader(
	test_data: list[dict[str, Any]],
	tokenizer: Any,
	batch_size: int = 8,
	num_workers: int = 0,
) -> DataLoader:
	"""Create an ordered test data loader that retains the final batch."""
	test_dataset = OrganizeBatch.InstructionDataset(test_data, tokenizer)

	return DataLoader(
		test_dataset,
		batch_size=batch_size,
		collate_fn=customized_collate_fn,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers,
	)


def print_train_loader_shapes(train_loader: DataLoader) -> None:
	"""Print the input and target tensor shapes for each training batch."""
	print("Train loader:")
	for inputs, targets in train_loader:
		print(inputs.shape, targets.shape)


def main() -> None:
	"""Create the training loader and print each batch's tensor shapes."""
	prepare_dataset = import_prepare_dataset()
	data = prepare_dataset.download_and_load_file(
		prepare_dataset.DATA_FILE, prepare_dataset.DATA_URL
	)
	train_data, _, _ = prepare_dataset.split_data(data)
	tokenizer = tiktoken.get_encoding("gpt2")
	train_loader = create_train_loader(train_data, tokenizer)
	print_train_loader_shapes(train_loader)


if __name__ == "__main__":
	main()


