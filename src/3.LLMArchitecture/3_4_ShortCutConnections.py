"""Demonstrate residual connections and their gradients."""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """Apply the Gaussian error linear unit approximation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU elementwise and return a tensor with the input shape."""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class ExampleDeepNeuralNetwork(nn.Module):
    """Stack feed-forward layers with optional residual shortcuts."""
    def __init__(self, layer_sizes: list[int], use_shortcut: bool) -> None:
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each layer and return its final two-dimensional output tensor."""
        for layer in self.layers:
            # Compute the output of the current layer.
            layer_output = layer(x)
            # Apply residual connection only when shapes match.
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model: nn.Module, x: torch.Tensor) -> None:
    """Print mean absolute weight gradients after one regression step."""
    model.zero_grad()

    # Forward pass.
    output = model(x)
    target = torch.tensor([[0.0]])

    # Calculate loss based on how close the target and output are.
    loss = nn.MSELoss()(output, target)

    # Backward pass to calculate gradients.
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} gradient mean abs: {param.grad.abs().mean().item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(123)

    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    print("Without shortcut:\n")
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)

    print("\nWith shortcut:\n")
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)
