from pathlib import Path
import tempfile
import unittest

import torch

from dataset import CorpusDataset
from main import collapse_adjacent_positions


class SimpleTokenizer:
    def tokenize(self, text):
        tokens = []
        current = []

        for char in text:
            if char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
                continue

            if char.isalnum():
                current.append(char)
                continue

            if current:
                tokens.append("".join(current))
                current = []

            tokens.append(char)

        if current:
            tokens.append("".join(current))

        return tokens


class CorpusDatasetTests(unittest.TestCase):
    def build_dataset(self, text):
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".sent_split",
            delete=False,
        ) as temp_file:
            temp_file.write(text)
            temp_path = Path(temp_file.name)

        self.addCleanup(temp_path.unlink)
        return CorpusDataset(temp_path, SimpleTokenizer(), window_size=5)

    def positive_positions(self, dataset):
        return [index for index, label in enumerate(dataset.corpus_labels) if label == 1]

    def test_all_tokens_are_candidate_positions(self):
        dataset = self.build_dataset("A test.<EOS> Next one<EOS>\nDone")

        self.assertEqual(list(dataset.candidate_positions), list(range(len(dataset.corpus_tokens))))
        self.assertEqual(len(dataset.corpus_labels), len(dataset.corpus_tokens))

    def test_training_sampler_keeps_all_positives_and_limits_negatives(self):
        dataset = self.build_dataset("A test.<EOS> Next one<EOS>\nDone")

        sampled_indices = dataset.sample_training_indices(
            negative_ratio=1,
            generator=torch.Generator().manual_seed(0),
        )
        sampled_labels = [dataset.corpus_labels[index] for index in sampled_indices]

        self.assertEqual(sum(sampled_labels), len(dataset.positive_indices))
        self.assertEqual(sampled_labels.count(0), len(dataset.positive_indices))
        self.assertTrue(set(dataset.positive_indices).issubset(set(sampled_indices)))

    def test_punctuation_cluster_marks_only_the_token_adjacent_to_eos(self):
        dataset = self.build_dataset("[Hello.]<EOS> Next")

        self.assertEqual(dataset.corpus_tokens, ["[", "Hello", ".", "]", "Next"])
        self.assertEqual(self.positive_positions(dataset), [3])
        self.assertEqual(dataset.target_stats["punctuation"], 1)

    def test_newline_anchor_is_used_when_sentence_has_no_final_punctuation(self):
        dataset = self.build_dataset("Hello world<EOS> \nNext line")

        self.assertEqual(dataset.corpus_tokens, ["Hello", "world", "\n", "Next", "line"])
        self.assertEqual(self.positive_positions(dataset), [2])
        self.assertEqual(dataset.target_stats["newline"], 1)

    def test_last_token_fallback_covers_unpunctuated_boundaries_without_newline(self):
        dataset = self.build_dataset("TEN YEARS ON<EOS> Next")

        self.assertEqual(dataset.corpus_tokens, ["TEN", "YEARS", "ON", "Next"])
        self.assertEqual(self.positive_positions(dataset), [2])
        self.assertEqual(dataset.target_stats["fallback"], 1)

    def test_sent_split_reader_handles_multiple_eos_on_the_same_line(self):
        dataset = self.build_dataset("First<EOS>Second line<EOS>\nThird one<EOS>")

        sentences = dataset.load_sent_split_sentences(Path(dataset.filename))
        self.assertEqual(sentences, ["First", "Second line", "Third one"])

    def test_inference_stream_does_not_glue_tokens_around_eos(self):
        dataset = self.build_dataset("Hello<EOS>World")

        inference_tokens, _ = dataset.build_inference_stream()
        self.assertEqual(inference_tokens, ["Hello", "World"])

    def test_adjacent_predicted_boundaries_are_collapsed_to_the_rightmost_position(self):
        positions = collapse_adjacent_positions([3, 4, 5, 8, 10, 11])
        self.assertEqual(positions, [5, 8, 11])


if __name__ == "__main__":
    unittest.main()
