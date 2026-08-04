"""Classify new text with the fine-tuned GPT spam classifier."""

import importlib.util
from pathlib import Path
from types import ModuleType

import tiktoken
import torch


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


def classify_text(
	text: str,
	model: torch.nn.Module,
	tokenizer: tiktoken.Encoding,
	device: torch.device,
	max_length: int,
	pad_token_id: int = 50256,
) -> str:
	"""Return ``spam`` or ``not spam`` for the supplied text."""
	if max_length <= 0:
		raise ValueError("max_length must be greater than zero")

	supported_context_length = model.pos_emb.weight.shape[0]
	effective_length = min(max_length, supported_context_length)
	input_ids = tokenizer.encode(text)[:effective_length]
	input_ids += [pad_token_id] * (effective_length - len(input_ids))
	inputs = torch.tensor(input_ids, device=device).unsqueeze(0)

	model.eval()
	with torch.no_grad():
		logits = model(inputs)[:, -1, :]
	predicted_label = torch.argmax(logits, dim=-1).item()
	return "spam" if predicted_label == 1 else "not spam"


def main() -> None:
	"""Load the fine-tuned classifier and classify a sample message."""
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
		f"Fine-tuned model not found at {MODEL_PATH}. Run 5.7.Finetuning.py first."
	)

	calc_loss = import_calc_loss()
	classification_head = calc_loss.import_classification_head()
	data_loaders = calc_loss.import_data_loaders()
	model, _ = classification_head.initialize_classifier_model()
	train_dataset, _, _ = data_loaders.create_datasets()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.load_state_dict(
		torch.load(MODEL_PATH, map_location=device, weights_only=True)
	)
	model.to(device)

	text1 = (
		"You are a winner you have been specially selected to receive "
		"$1000 cash or a $2000 award."
	)
	text2 = (
            "You need to set goals to be successful."
        )
	tokenizer = tiktoken.get_encoding("gpt2")
	max_length = train_dataset.max_length
	print(classify_text(text1, model, tokenizer, device, max_length))
	print(classify_text(text2, model, tokenizer, device, max_length))


if __name__ == "__main__":
	main()
