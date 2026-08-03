import unittest

import torch
from torch import nn

from module_loader import load_module


previous_chapters = load_module(
    "previous_chapters", "src/4.Training/PreviousChapters.py"
)


class StubTokenizer:
    def encode(self, text, allowed_special=None):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class RecordingModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs = []

    def forward(self, token_ids):
        self.inputs.append(token_ids.clone())
        logits = torch.zeros(
            token_ids.shape[0], token_ids.shape[1], self.vocab_size
        )
        logits[:, -1, 3] = 1
        return logits


class PreviousChaptersTests(unittest.TestCase):
    def test_dataset_builds_shifted_sliding_windows(self):
        dataset = previous_chapters.GPTDatasetV1(
            "abcdef", StubTokenizer(), max_length=3, stride=2
        )

        self.assertEqual(len(dataset), 2)
        first_input, first_target = dataset[0]
        second_input, second_target = dataset[1]
        self.assertTrue(torch.equal(first_input, torch.tensor([97, 98, 99])))
        self.assertTrue(torch.equal(first_target, torch.tensor([98, 99, 100])))
        self.assertTrue(torch.equal(second_input, torch.tensor([99, 100, 101])))
        self.assertTrue(torch.equal(second_target, torch.tensor([100, 101, 102])))

    def test_layer_norm_normalizes_each_token(self):
        layer_norm = previous_chapters.LayerNorm(3)
        inputs = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])

        output = layer_norm(inputs)

        self.assertTrue(torch.allclose(output.mean(dim=-1), torch.zeros(1, 2)))
        self.assertTrue(
            torch.allclose(
                output.var(dim=-1, unbiased=False),
                torch.ones(1, 2),
                atol=2e-5,
            )
        )

    def test_multi_head_attention_preserves_batch_and_sequence_shape(self):
        attention = previous_chapters.MultiHeadAttention(
            d_in=4,
            d_out=6,
            context_length=5,
            dropout=0.0,
            num_heads=2,
        )

        output = attention(torch.randn(2, 5, 4))

        self.assertEqual(output.shape, (2, 5, 6))

    def test_gpt_model_returns_logits_for_each_input_token(self):
        config = {
            "vocab_size": 11,
            "context_length": 6,
            "emb_dim": 8,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
        model = previous_chapters.GPTModel(config)

        logits = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(logits.shape, (2, 3, 11))

    def test_generation_crops_context_and_appends_greedy_tokens(self):
        model = RecordingModel(vocab_size=5)
        initial_tokens = torch.tensor([[0, 1, 2]])

        generated = previous_chapters.generate_text_simple(
            model, initial_tokens, max_new_tokens=2, context_size=2
        )

        self.assertTrue(torch.equal(generated, torch.tensor([[0, 1, 2, 3, 3]])))
        self.assertTrue(torch.equal(model.inputs[0], torch.tensor([[1, 2]])))
        self.assertTrue(torch.equal(model.inputs[1], torch.tensor([[2, 3]])))

    def test_token_id_helpers_round_trip_text(self):
        tokenizer = StubTokenizer()

        token_ids = previous_chapters.text_to_token_ids("test", tokenizer)

        self.assertEqual(token_ids.shape, (1, 4))
        self.assertEqual(
            previous_chapters.token_ids_to_text(token_ids, tokenizer), "test"
        )


if __name__ == "__main__":
    unittest.main()
