from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PAD_TOKEN_ID = 50256
LABEL_IDS = {"ham": 0, "spam": 1}
BATCH_SIZE = 8
NUM_WORKERS = 0


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id: int = PAD_TOKEN_ID,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"].astype(str)
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            if max_length <= 0:
                raise ValueError("max_length must be greater than zero")
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text
            + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoded_texts[index]
        label = LABEL_IDS[self.data.iloc[index]["Label"]]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        if not self.encoded_texts:
            raise ValueError("Dataset must contain at least one text")
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_datasets() -> tuple[SpamDataset, SpamDataset, SpamDataset]:
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(DATA_DIR / "train.csv", tokenizer)
    validation_dataset = SpamDataset(
        DATA_DIR / "validation.csv", tokenizer, max_length=train_dataset.max_length
    )
    test_dataset = SpamDataset(
        DATA_DIR / "test.csv", tokenizer, max_length=train_dataset.max_length
    )
    return train_dataset, validation_dataset, test_dataset


def create_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, validation_dataset, test_dataset = create_datasets()
    torch.manual_seed(123)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = create_data_loaders()
    input_batch, target_batch = next(iter(train_loader))
    print("Train loader:")
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions:", target_batch.shape)
    print(
        "Batch counts: "
        f"train={len(train_loader)}, "
        f"validation={len(validation_loader)}, "
        f"test={len(test_loader)}"
    )
