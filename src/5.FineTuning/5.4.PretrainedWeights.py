"""Initialize a pretrained GPT-2 model for fine-tuning."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import tiktoken
import torch


TRAINING_DIR = Path(__file__).resolve().parents[1] / "4.Training"
CHOOSE_MODEL = "gpt2-small"


def import_training_module(module_name: str, filename: str) -> ModuleType:
    """Load and return a Python module from the training directory.

        Raises:
            ImportError: If an import specification or loader cannot be created.
        """
    module_path = TRAINING_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PREVIOUS_CHAPTERS = import_training_module("PreviousChapters", "PreviousChapters.py")


def initialize_pretrained_model() -> tuple[torch.nn.Module, dict[str, object]]:
    """Return a pretrained GPT-2 model and its local configuration."""
    weight_loader = import_training_module(
        "load_openai_weights", "4.5.LoadOpenAIWeights.py"
    )
    model, config = weight_loader.load_pretrained_gpt2(CHOOSE_MODEL)
    return model, config


def generate_text(model: torch.nn.Module, config: dict[str, object]) -> str:
    """Generate and return sample text with a pretrained model.

        Raises:
            TypeError: If the configured context length is not an integer.
        """
    tokenizer = tiktoken.get_encoding("gpt2")
    context_length = config["context_length"]
    if not isinstance(context_length, int):
        raise TypeError("context_length must be an integer")

    token_ids = PREVIOUS_CHAPTERS.generate_text_simple(
        model=model,
        idx=PREVIOUS_CHAPTERS.text_to_token_ids(
            "Every effort moves you", tokenizer
        ),
        max_new_tokens=15,
        context_size=context_length,
    )
    return PREVIOUS_CHAPTERS.token_ids_to_text(token_ids, tokenizer)


if __name__ == "__main__":
    torch.manual_seed(123)
    model, config = initialize_pretrained_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {CHOOSE_MODEL} with {parameter_count:,} parameters")
    print("Model configuration:", config)
    print("Generated text:", generate_text(model, config))
