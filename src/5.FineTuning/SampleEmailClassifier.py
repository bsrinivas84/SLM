import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_emails.csv"
CONFUSION_MATRIX_FILE = (
    Path(__file__).resolve().parents[2] / "Images" / "email_confusion_matrix.png"
)
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def load_emails(path: Path) -> tuple[list[str], list[int]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows or not {"target", "email"}.issubset(rows[0]):
        raise ValueError("Dataset must contain target and email columns")

    return [row["email"] for row in rows], [int(row["target"]) for row in rows]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(emails: list[str]) -> dict[str, int]:
    token_counts = Counter(token for email in emails for token in tokenize(email))
    return {
        token: index
        for index, (token, _) in enumerate(sorted(token_counts.items()))
    }


def vectorize(emails: list[str], vocabulary: dict[str, int]) -> torch.Tensor:
    vectors = torch.zeros((len(emails), len(vocabulary)), dtype=torch.float32)
    for row, email in enumerate(emails):
        for token in set(tokenize(email)):
            if token in vocabulary:
                vectors[row, vocabulary[token]] = 1.0
    return vectors


def stratified_split(labels: list[int]) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    class_counts = Counter()

    for index, label in enumerate(labels):
        class_counts[label] += 1
        destination = validation_indices if class_counts[label] % 5 == 0 else train_indices
        destination.append(index)
    return train_indices, validation_indices


def evaluate(
    model: nn.Module, features: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_classifier(
    emails: list[str], labels: list[int], epochs: int
) -> tuple[nn.Module, dict[str, int], torch.Tensor, torch.Tensor]:
    train_indices, validation_indices = stratified_split(labels)
    train_emails = [emails[index] for index in train_indices]
    vocabulary = build_vocabulary(train_emails)
    features = vectorize(emails, vocabulary)
    targets = torch.tensor(labels, dtype=torch.long)

    train_features = features[train_indices]
    train_targets = targets[train_indices]
    validation_features = features[validation_indices]
    validation_targets = targets[validation_indices]
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=8, shuffle=True
    )

    model = nn.Sequential(
        nn.Linear(len(vocabulary), 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

    train_loss, train_accuracy = evaluate(model, train_features, train_targets, criterion)
    validation_loss, validation_accuracy = evaluate(
        model, validation_features, validation_targets, criterion
    )
    print(f"Loaded {len(emails)} emails; vocabulary size: {len(vocabulary)}")
    print(f"Train      loss: {train_loss:.4f}, accuracy: {train_accuracy:.1%}")
    print(f"Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy:.1%}")
    return model, vocabulary, validation_features, validation_targets


def save_confusion_matrix(
    model: nn.Module, features: torch.Tensor, targets: torch.Tensor, path: Path
) -> None:
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)

    matrix = torch.bincount(targets * 2 + predictions, minlength=4).reshape(2, 2)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Email Classifier Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not spam", "Spam"],
        yticklabels=["Not spam", "Spam"],
    )

    threshold = matrix.max().item() / 2
    for true_label in range(2):
        for predicted_label in range(2):
            count = matrix[true_label, predicted_label].item()
            axis.text(
                predicted_label,
                true_label,
                str(count),
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix ({int(matrix.sum())} samples):\n{matrix.numpy()}")
    print(f"Saved confusion matrix to {path}")


def classify(model: nn.Module, vocabulary: dict[str, int], email: str) -> None:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(vectorize([email], vocabulary)), dim=1)[0]
    prediction = int(probabilities.argmax().item())
    label = "spam" if prediction == 1 else "not spam"
    print(f"{label:8} ({probabilities[prediction]:.1%}): {email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sample email spam classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--email", help="Optional email text to classify after training")
    args = parser.parse_args()

    torch.manual_seed(42)
    emails, labels = load_emails(DATA_FILE)
    model, vocabulary, validation_features, validation_targets = train_classifier(
        emails, labels, args.epochs
    )
    save_confusion_matrix(
        model, validation_features, validation_targets, CONFUSION_MATRIX_FILE
    )

    samples = [args.email] if args.email else [
        "Please review the meeting notes before our call tomorrow.",
        "Claim your free cash prize now by clicking this urgent link!",
    ]
    print("\nPredictions:")
    for email in samples:
        classify(model, vocabulary, email)


if __name__ == "__main__":
    main()
