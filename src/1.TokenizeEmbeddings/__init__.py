"""Shared device selection helpers for tokenization examples."""

import torch


def get_default_device() -> torch.device:
    """Return the CUDA device when available, otherwise the CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_default_device()
