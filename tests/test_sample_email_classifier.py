import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from module_loader import load_module


classifier = load_module(
    "sample_email_classifier", "src/5.FineTuning/SampleEmailClassifier.py"
)


class SampleEmailClassifierTests(unittest.TestCase):
    def test_tokenize_normalizes_case_and_keeps_apostrophes(self):
        self.assertEqual(
            classifier.tokenize("Don't MISS this: Offer #2!"),
            ["don't", "miss", "this", "offer", "2"],
        )

    def test_vocabulary_is_unique_and_alphabetically_ordered(self):
        vocabulary = classifier.build_vocabulary(["Beta alpha alpha", "Gamma"])

        self.assertEqual(vocabulary, {"alpha": 0, "beta": 1, "gamma": 2})

    def test_vectorize_creates_binary_bag_of_words(self):
        vocabulary = {"alpha": 0, "beta": 1}

        vectors = classifier.vectorize(
            ["Alpha alpha unknown", "beta alpha"], vocabulary
        )

        expected = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        self.assertTrue(torch.equal(vectors, expected))

    def test_stratified_split_sends_every_fifth_class_item_to_validation(self):
        labels = [0, 1] * 5

        train, validation = classifier.stratified_split(labels)

        self.assertEqual(validation, [8, 9])
        self.assertEqual(train, list(range(8)))

    def test_load_emails_reads_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emails.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["target", "email"])
                writer.writeheader()
                writer.writerow({"target": "1", "email": "Prize"})

            emails, labels = classifier.load_emails(path)

            self.assertEqual(emails, ["Prize"])
            self.assertEqual(labels, [1])

    def test_load_emails_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emails.csv"
            path.write_text("message\nhello\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "Dataset must contain target and email columns"
            ):
                classifier.load_emails(path)

    def test_evaluate_returns_cross_entropy_and_accuracy(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        features = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        labels = torch.tensor([0, 1])

        loss, accuracy = classifier.evaluate(
            model, features, labels, nn.CrossEntropyLoss()
        )

        self.assertAlmostEqual(loss, 0.126928, places=5)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
