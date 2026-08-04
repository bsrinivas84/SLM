"""Calculate classification loss for the fine-tuned GPT model."""

import importlib.util
from pathlib import Path
from types import ModuleType

import tiktoken
import torch
from torch.utils.data import DataLoader


def import_classification_head() -> ModuleType:
	"""Load and return the neighboring classification-head module."""
	module_path = Path(__file__).with_name("5.5.ClassificationHead.py")
	spec = importlib.util.spec_from_file_location("classification_head", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def import_data_loaders() -> ModuleType:
	"""Load and return the neighboring data-loader module."""
	module_path = Path(__file__).with_name("5.3.DataLoaders.py")
	spec = importlib.util.spec_from_file_location("data_loaders", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def calc_accuracy_loader(
	data_loader: DataLoader,
	model: torch.nn.Module,
	device: torch.device,
	num_batches: int | None = None,
) -> float:
	"""Return classification accuracy for the requested number of batches."""
	model.eval()
	correct_predictions = 0
	num_examples = 0

	if num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))

	with torch.no_grad():
		for batch_index, (input_batch, target_batch) in enumerate(data_loader):
			if batch_index >= num_batches:
				break

			input_batch = input_batch.to(device)
			target_batch = target_batch.to(device)
			logits = model(input_batch)[:, -1, :]
			predicted_labels = torch.argmax(logits, dim=-1)
			num_examples += predicted_labels.shape[0]
			correct_predictions += (predicted_labels == target_batch).sum().item()

	if num_examples == 0:
		raise ValueError("Cannot calculate accuracy without any examples")
	return correct_predictions / num_examples


def calc_loss_batch(
	input_batch: torch.Tensor,
	target_batch: torch.Tensor,
	model: torch.nn.Module,
	device: torch.device,
) -> torch.Tensor:
	"""Return cross-entropy loss for one classification batch."""
	input_batch = input_batch.to(device)
	target_batch = target_batch.to(device)
	logits = model(input_batch)[:, -1, :]
	return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(
	data_loader: DataLoader,
	model: torch.nn.Module,
	device: torch.device,
	num_batches: int | None = None,
) -> float:
	"""Return the mean classification loss across the requested batches."""
	if len(data_loader) == 0:
		return float("nan")

	if num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))

	if num_batches <= 0:
		return float("nan")

	total_loss = 0.0
	model.eval()
	with torch.no_grad():
		for batch_index, (input_batch, target_batch) in enumerate(data_loader):
			if batch_index >= num_batches:
				break
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			total_loss += loss.item()

	return total_loss / num_batches


def main() -> None:
	"""Initialize the GPT classification model."""
	classification_head = import_classification_head()
	data_loaders = import_data_loaders()
	model, config = classification_head.initialize_classifier_model()
	print("Classification model initialized")
	print("Model configuration:", config)
	print("Output head:", model.out_head)

	print("Output head:", model.out_head)

	tokenizer = tiktoken.get_encoding("gpt2")
	inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
	print("Inputs:", inputs)
	print("Input dimensions:", inputs.shape)

	# The above lines are now correctly indented within the main function.
	model.eval()
	with torch.no_grad():
		outputs = model(inputs)
		print("Outputs:", outputs)
		print("Output dimensions:", outputs.shape)
		print("Last logit",outputs[:,-1,:])
	# Calculate classification loss
	logits = outputs[:,-1,:]
	labels = torch.argmax(logits, dim=-1)
	loss_fn = torch.nn.CrossEntropyLoss()
	loss = loss_fn(logits, labels)
	print("Classification loss:", loss)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	torch.manual_seed(123)
	train_loader, validation_loader, test_loader = data_loaders.create_data_loaders()
	train_accuracy = calc_accuracy_loader(
		train_loader, model, device, num_batches=10
	)
	validation_accuracy = calc_accuracy_loader(
		validation_loader, model, device, num_batches=10
	)
	test_accuracy = calc_accuracy_loader(
		test_loader, model, device, num_batches=10
	)
	train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
	validation_loss = calc_loss_loader(
		validation_loader, model, device, num_batches=10
	)
	test_loss = calc_loss_loader(test_loader, model, device, num_batches=10)
	print(f"Training accuracy: {train_accuracy:.2%}")
	print(f"Validation accuracy: {validation_accuracy:.2%}")
	print(f"Test accuracy: {test_accuracy:.2%}")
	print(f"Training loss: {train_loss:.3f}")
	print(f"Validation loss: {validation_loss:.3f}")
	print(f"Test loss: {test_loss:.3f}")

  

if __name__ == "__main__":
	main()
