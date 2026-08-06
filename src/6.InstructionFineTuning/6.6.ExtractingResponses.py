"""Generate and save responses for the instruction test split."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import tiktoken
import torch
from tqdm import tqdm


OUTPUT_PATH = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "raw"
	/ "instruction-data-with-response.json"
)


def import_neighbor(module_name: str, filename: str) -> ModuleType:
	"""Load a module from the instruction fine-tuning chapter."""
	module_path = Path(__file__).with_name(filename)
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def add_model_responses(
	test_data: list[dict[str, Any]],
	model: torch.nn.Module,
	config: dict[str, int | float | bool],
	tokenizer: tiktoken.Encoding,
	prepare_dataset: ModuleType,
	pretrained_model: ModuleType,
) -> list[dict[str, Any]]:
	"""Generate and attach a model response to every test entry."""
	torch.manual_seed(123)
	entries_with_responses = [entry.copy() for entry in test_data]

	for entry in tqdm(entries_with_responses, desc="Generating responses"):
		input_text = prepare_dataset.format_input(entry)
		generated_text = pretrained_model.generate_text(
			model,
			config,
			input_text,
			tokenizer,
			max_new_tokens=256,
		)
		entry["model_response"] = (
			generated_text[len(input_text) :]
			.replace("### Response:", "")
			.strip()
		)

	return entries_with_responses


def save_responses(entries: list[dict[str, Any]], output_path: Path) -> None:
	"""Write generated test entries as pretty-printed JSON."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as file:
		json.dump(entries, file, indent=4, ensure_ascii=False)


def main() -> None:
	"""Load the fine-tuned model and save responses for the test split."""
	prepare_dataset = import_neighbor("prepare_dataset", "6.1.PrepareDataset.py")
	pretrained_model = import_neighbor(
		"load_pretrained_model", "6.4.LoadPretrainedModel.py"
	)
	fine_tuning = import_neighbor(
		"fine_tuning_instruction", "6.5.FineTuningInstruction.py"
	)

	if not fine_tuning.CHECKPOINT_PATH.exists():
		raise FileNotFoundError(
			f"Fine-tuned checkpoint not found: {fine_tuning.CHECKPOINT_PATH}. "
			"Run 6.5.FineTuningInstruction.py first."
		)

	data = prepare_dataset.download_and_load_file(
		prepare_dataset.DATA_FILE, prepare_dataset.DATA_URL
	)
	_, test_data, _ = prepare_dataset.split_data(data)
	tokenizer = tiktoken.get_encoding("gpt2")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, config = fine_tuning.load_finetuned_model(
		pretrained_model, fine_tuning.CHECKPOINT_PATH, device
	)

	entries_with_responses = add_model_responses(
		test_data,
		model,
		config,
		tokenizer,
		prepare_dataset,
		pretrained_model,
	)
	save_responses(entries_with_responses, OUTPUT_PATH)
	print(f"Responses saved to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
