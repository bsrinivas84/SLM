"""Load a pretrained GPT-2 model for instruction fine-tuning."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import tiktoken
import torch


TRAINING_DIR = Path(__file__).resolve().parents[1] / "4.Training"
BASE_CONFIG: dict[str, int | float | bool] = {
	"vocab_size": 50257,
	"context_length": 1024,
	"drop_rate": 0.0,
	"qkv_bias": True,
}
MODEL_CONFIGS = {
	"gpt2-small (124M)": {
		"model_name": "gpt2-small",
		"emb_dim": 768,
		"n_layers": 12,
		"n_heads": 12,
	},
	"gpt2-medium (355M)": {
		"model_name": "gpt2-medium",
		"emb_dim": 1024,
		"n_layers": 24,
		"n_heads": 16,
	},
	"gpt2-large (774M)": {
		"model_name": "gpt2-large",
		"emb_dim": 1280,
		"n_layers": 36,
		"n_heads": 20,
	},
	"gpt2-xl (1558M)": {
		"model_name": "gpt2-xl",
		"emb_dim": 1600,
		"n_layers": 48,
		"n_heads": 25,
	},
}
CHOOSE_MODEL = "gpt2-medium (355M)"


def import_training_module(module_name: str, filename: str) -> ModuleType:
	"""Load a module from the training chapter."""
	module_path = TRAINING_DIR / filename
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def load_pretrained_model(
	chosen_model: str = CHOOSE_MODEL,
) -> tuple[torch.nn.Module, dict[str, int | float | bool]]:
	"""Load the selected pretrained GPT-2 model and return its configuration."""
	if chosen_model not in MODEL_CONFIGS:
		choices = ", ".join(MODEL_CONFIGS)
		raise ValueError(f"Unknown model {chosen_model!r}. Choose from: {choices}")

	model_settings = MODEL_CONFIGS[chosen_model]
	model_name = model_settings["model_name"]
	if not isinstance(model_name, str):
		raise TypeError("model_name must be a string")

	import_training_module("PreviousChapters", "PreviousChapters.py")
	weight_loader = import_training_module(
		"load_openai_weights", "4.5.LoadOpenAIWeights.py"
	)
	model, loaded_config = weight_loader.load_pretrained_gpt2(model_name)

	config = BASE_CONFIG.copy()
	config.update(
		{
			key: value
			for key, value in model_settings.items()
			if key != "model_name"
		}
	)
	config.update(loaded_config)
	config["drop_rate"] = BASE_CONFIG["drop_rate"]
	model.eval()
	return model, config


def generate_text(
	model: torch.nn.Module,
	config: dict[str, int | float | bool],
	input_text: str,
	tokenizer: tiktoken.Encoding,
) -> str:
	"""Generate and decode a response from the pretrained model."""
	context_length = config["context_length"]
	if not isinstance(context_length, int):
		raise TypeError("context_length must be an integer")

	previous_chapters = import_training_module(
		"PreviousChapters", "PreviousChapters.py"
	)
	generation = import_training_module(
		"use_top_k_temperature", "4.3.3.UseTopKTemp.py"
	)
	device = next(model.parameters()).device
	token_ids = generation.generate_text_simple(
		model=model,
		idx=previous_chapters.text_to_token_ids(input_text, tokenizer).to(device),
		max_new_tokens=35,
		context_size=context_length,
		eos_token_id=50256,
	)
	return previous_chapters.token_ids_to_text(token_ids.cpu(), tokenizer)


def main() -> None:
	"""Load the configured model and generate text for a sample instruction."""
	model, config = load_pretrained_model()

	tokenizer = tiktoken.get_encoding("gpt2")
	input_text = (
		"Below is an instruction that describes a task. "
		"Write a response that appropriately completes the request."
		"\n\n### Instruction:\n Pluralize the following sentence: "
		"'Apple Tree has an apple.'"
		"\n\n### Response:\n"
	)
	generated_text = generate_text(model, config, input_text, tokenizer)
	print(generated_text)


if __name__ == "__main__":
	main()
