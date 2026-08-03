import unittest

import pandas as pd

from module_loader import load_module


prepare_dataset = load_module(
    "prepare_dataset", "src/5.FineTuning/5.1.PrepareDataset.py"
)


class PrepareDatasetTests(unittest.TestCase):
    def test_balanced_dataset_matches_minority_class_size(self):
        dataframe = pd.DataFrame(
            {
                "Label": ["ham", "ham", "ham", "spam", "spam"],
                "Text": ["h1", "h2", "h3", "s1", "s2"],
            }
        )

        balanced = prepare_dataset.create_balanced_dataset(dataframe)

        self.assertEqual(balanced["Label"].value_counts().to_dict(), {"ham": 2, "spam": 2})
        self.assertEqual(balanced.index.tolist(), [0, 1, 2, 3])

    def test_balancing_is_deterministic(self):
        dataframe = pd.DataFrame(
            {
                "Label": ["ham", "ham", "ham", "spam"],
                "Text": ["h1", "h2", "h3", "s1"],
            }
        )

        first = prepare_dataset.create_balanced_dataset(dataframe)
        second = prepare_dataset.create_balanced_dataset(dataframe)

        pd.testing.assert_frame_equal(first, second)

    def test_random_split_has_expected_sizes_and_no_overlap(self):
        dataframe = pd.DataFrame(
            {"Label": ["ham"] * 10, "Text": [f"message-{index}" for index in range(10)]}
        )

        train, validation, test = prepare_dataset.random_split(
            dataframe, train_fraction=0.6, validation_fraction=0.2
        )

        self.assertEqual((len(train), len(validation), len(test)), (6, 2, 2))
        split_messages = set(train["Text"]) | set(validation["Text"]) | set(test["Text"])
        self.assertEqual(split_messages, set(dataframe["Text"]))
        self.assertFalse(set(train["Text"]) & set(validation["Text"]))
        self.assertFalse(set(train["Text"]) & set(test["Text"]))
        self.assertFalse(set(validation["Text"]) & set(test["Text"]))


if __name__ == "__main__":
    unittest.main()
