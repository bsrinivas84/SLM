"""Evaluate a pretrained GPT-2 model on instruction data."""

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import tiktoken
import torch


TRAINING_DIR = Path(__file__).resolve().parents[1] / "4.Training"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "models"
    / "instruction-finetuned-gpt2-medium.pth"
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


import_training_module("PreviousChapters", "PreviousChapters.py")
training = import_training_module("training", "4.2.Training.py")
calc_loss_loader = training.calc_loss_loader
train_model_simple = training.train_model_simple


def save_finetuned_model(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Save fine-tuned model weights to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_finetuned_model(
    pretrained_model: ModuleType,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int | float | bool]]:
    """Create a compatible GPT-2 model and load fine-tuned weights."""
    model, config = pretrained_model.load_pretrained_model()
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def test_finetuned_model(
    model: torch.nn.Module,
    config: dict[str, int | float | bool],
    test_data: list[dict[str, Any]],
    tokenizer: tiktoken.Encoding,
    prepare_dataset: ModuleType,
    pretrained_model: ModuleType,
) -> None:
    """Print expected and generated responses for three test examples."""
    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = prepare_dataset.format_input(entry)
        generated_text = pretrained_model.generate_text(
            model,
            config,
            input_text,
            tokenizer,
            max_new_tokens=256,
        )
        response_text = (
            generated_text[len(input_text) :]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text}")
        print("-" * 50)


def main() -> None:
    """Calculate initial training and validation losses over five batches."""
    prepare_dataset = import_neighbor("prepare_dataset", "6.1.PrepareDataset.py")
    data_loaders = import_neighbor("instruction_data_loaders", "6.3.DataLoader.py")
    pretrained_model = import_neighbor(
        "load_pretrained_model", "6.4.LoadPretrainedModel.py"
    )

    data = prepare_dataset.download_and_load_file(
        prepare_dataset.DATA_FILE, prepare_dataset.DATA_URL
    )
    train_data, test_data, val_data = prepare_dataset.split_data(data)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader = data_loaders.create_train_loader(train_data, tokenizer)
    val_loader = data_loaders.create_val_loader(val_data, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = pretrained_model.load_pretrained_model()
    model.to(device)
    torch.manual_seed(123)

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context=prepare_dataset.format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    save_finetuned_model(model, CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")

    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loaded_model, config = load_finetuned_model(
        pretrained_model, CHECKPOINT_PATH, device
    )
    test_finetuned_model(
        loaded_model,
        config,
        test_data,
        tokenizer,
        prepare_dataset,
        pretrained_model,
    )


if __name__ == "__main__":
    main()