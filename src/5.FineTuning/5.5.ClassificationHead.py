"""Replace a pretrained GPT model output head for classification."""

import importlib.util
from pathlib import Path
from types import ModuleType

import tiktoken
import torch


def import_pretrained_weights() -> ModuleType:
    """Load and return the neighboring pretrained-weight module.

        Raises:
            ImportError: If an import specification or loader cannot be created.
        """
    module_path = Path(__file__).with_name("5.4.PretrainedWeights.py")
    spec = importlib.util.spec_from_file_location("pretrained_weights", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_classifier_model(
    num_classes: int = 2,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Return a pretrained GPT model configured for classification."""
    pretrained_weights = import_pretrained_weights()
    model, config = pretrained_weights.initialize_pretrained_model()

    for parameter in model.parameters():
        parameter.requires_grad = False

    embedding_dimension = config["emb_dim"]
    if not isinstance(embedding_dimension, int):
        raise TypeError("emb_dim must be an integer")
    model.out_head = torch.nn.Linear(embedding_dimension, num_classes)

    for parameter in model.trf_blocks[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True

    return model, config


def main() -> None:
    """Run a sample forward pass with the classification model."""
    model, _ = initialize_classifier_model()
    tokenizer = tiktoken.get_encoding("gpt2")
    inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    print("Inputs:", inputs)
    print("Input dimensions:", inputs.shape)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:", outputs)
    print("Output dimensions:", outputs.shape)


if __name__ == "__main__":
    main()

