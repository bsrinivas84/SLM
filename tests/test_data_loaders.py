import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from module_loader import load_module


data_loaders = load_module("data_loaders", "src/5.FineTuning/5.3.DataLoaders.py")


class StubTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]


class SpamDatasetTests(unittest.TestCase):
    def write_csv(self, directory, rows):
        path = Path(directory) / "messages.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_dataset_pads_to_longest_encoded_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                [
                    {"Label": "ham", "Text": "ab"},
                    {"Label": "spam", "Text": "x"},
                ],
            )

            dataset = data_loaders.SpamDataset(path, StubTokenizer(), pad_token_id=0)

            first_text, first_label = dataset[0]
            second_text, second_label = dataset[1]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.max_length, 2)
            self.assertTrue(torch.equal(first_text, torch.tensor([97, 98])))
            self.assertTrue(torch.equal(second_text, torch.tensor([120, 0])))
            self.assertEqual((first_label.item(), second_label.item()), (0, 1))

    def test_dataset_truncates_to_explicit_max_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory, [{"Label": "ham", "Text": "abcdef"}]
            )

            dataset = data_loaders.SpamDataset(
                path, StubTokenizer(), max_length=3, pad_token_id=0
            )

            encoded, _ = dataset[0]
            self.assertTrue(torch.equal(encoded, torch.tensor([97, 98, 99])))

    def test_dataset_rejects_non_positive_max_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [{"Label": "ham", "Text": "text"}])

            with self.assertRaisesRegex(
                ValueError, "max_length must be greater than zero"
            ):
                data_loaders.SpamDataset(path, StubTokenizer(), max_length=0)

    def test_dataset_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, {"Label": [], "Text": []})

            with self.assertRaisesRegex(
                ValueError, "Dataset must contain at least one text"
            ):
                data_loaders.SpamDataset(path, StubTokenizer())


if __name__ == "__main__":
    unittest.main()
