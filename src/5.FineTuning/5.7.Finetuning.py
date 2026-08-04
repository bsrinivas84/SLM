"""Fine-tune the GPT model for spam classification."""

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.data import DataLoader


MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "spam_classifier.pth"


def import_calc_loss() -> ModuleType:
	"""Load and return the neighboring loss-calculation module."""
	module_path = Path(__file__).with_name("5.6.CalcLoss.py")
	spec = importlib.util.spec_from_file_location("calc_loss", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def evaluate_model(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	device: torch.device,
	eval_iter: int,
	calc_loss: ModuleType,
) -> tuple[float, float]:
	"""Return training and validation loss over a limited number of batches."""
	model.eval()
	with torch.no_grad():
		train_loss = calc_loss.calc_loss_loader(
			train_loader, model, device, num_batches=eval_iter
		)
		validation_loss = calc_loss.calc_loss_loader(
			validation_loader, model, device, num_batches=eval_iter
		)
	model.train()
	return train_loss, validation_loss


def train_classifier_simple(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	num_epochs: int,
	eval_freq: int,
	eval_iter: int,
	calc_loss: ModuleType,
) -> tuple[list[float], list[float], list[float], list[float], int]:
	"""Train the classifier and return its loss and accuracy history."""
	train_losses: list[float] = []
	validation_losses: list[float] = []
	train_accuracies: list[float] = []
	validation_accuracies: list[float] = []
	examples_seen = 0
	global_step = -1

	for epoch in range(num_epochs):
		model.train()
		for input_batch, target_batch in train_loader:
			optimizer.zero_grad()
			loss = calc_loss.calc_loss_batch(
				input_batch, target_batch, model, device
			)
			loss.backward()
			optimizer.step()
			examples_seen += input_batch.shape[0]
			global_step += 1

			if global_step % eval_freq == 0:
				train_loss, validation_loss = evaluate_model(
					model,
					train_loader,
					validation_loader,
					device,
					eval_iter,
					calc_loss,
				)
				train_losses.append(train_loss)
				validation_losses.append(validation_loss)
				print(
					f"Ep {epoch + 1} (Step {global_step:06d}): "
					f"Train loss {train_loss:.3f}, "
					f"Val loss {validation_loss:.3f}"
				)

		train_accuracy = calc_loss.calc_accuracy_loader(
			train_loader, model, device, num_batches=eval_iter
		)
		validation_accuracy = calc_loss.calc_accuracy_loader(
			validation_loader, model, device, num_batches=eval_iter
		)
		print(
			f"Training accuracy: {train_accuracy * 100:.2f}% | "
			f"Validation accuracy: {validation_accuracy * 100:.2f}%"
		)
		train_accuracies.append(train_accuracy)
		validation_accuracies.append(validation_accuracy)

	return (
		train_losses,
		validation_losses,
		train_accuracies,
		validation_accuracies,
		examples_seen,
	)


def main() -> None:
	"""Initialize and fine-tune the spam classifier."""
	torch.manual_seed(123)
	calc_loss = import_calc_loss()
	classification_head = calc_loss.import_classification_head()
	data_loaders = calc_loss.import_data_loaders()
	model, _ = classification_head.initialize_classifier_model()
	train_loader, validation_loader, _ = data_loaders.create_data_loaders()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=5e-5, weight_decay=0.1
	)

	train_classifier_simple(
		model=model,
		train_loader=train_loader,
		validation_loader=validation_loader,
		optimizer=optimizer,
		device=device,
		num_epochs=5,
		eval_freq=50,
		eval_iter=5,
		calc_loss=calc_loss,
	)
	torch.save(model.state_dict(), MODEL_PATH)
	print(f"Saved fine-tuned model to {MODEL_PATH}")


if __name__ == "__main__":
	main()
