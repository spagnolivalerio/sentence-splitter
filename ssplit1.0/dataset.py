from pathlib import Path
import re
import unicodedata
import torch
from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    def __init__(
        self,
        filename,
        tokenizer,
        window_size=13,
        ignore_index=255,
        token_to_id=None,
    ):
        self.filename = str(self.resolve_dataset_path(filename))
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.ignore_index = ignore_index
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.newline_token = "\n"
        self.excluded_delimiters = {
            ",",
            "(",
            "[",
            "{",
            "<",
            "#",
            "%",
            "&",
            "^",
            "£",
            "$",
            "/",
            "=",
            "\\",
            "|",
            "-",
        }

        (
            self.corpus_tokens,
            self.corpus_sentence_mask,
            self.corpus_labels,
        ) = self.build_corpus_targets()
        if token_to_id is None:
            self.token_to_id, self.id_to_token = self.build_vocabulary()
        else:
            self.token_to_id = token_to_id
            self.id_to_token = {i: token for token, i in token_to_id.items()}
        self.corpus_token_ids = self.encode_corpus_tokens()
        self.candidate_positions = self.build_candidate_positions()

    @staticmethod
    def resolve_dataset_path(filename):
        """
        Resolve dataset paths in the current project layout.
        """
        dataset_path = Path(filename)
        if dataset_path.exists():
            return dataset_path

        project_root = Path(__file__).resolve().parent
        direct_candidate = project_root / dataset_path
        if direct_candidate.exists():
            return direct_candidate

        exact_matches = sorted(project_root.rglob(dataset_path.name))
        if exact_matches:
            return exact_matches[0]

        stem_matches = sorted(project_root.rglob(f"{dataset_path.stem}.*"))
        supported_matches = [path for path in stem_matches if path.suffix == ".sent_split"]
        if supported_matches:
            return supported_matches[0]

        raise FileNotFoundError(f"Dataset file not found: {filename}")

    def load_sent_split_sentences(self, dataset_path):
        sentences = []

        with dataset_path.open("r", encoding="utf-8") as file:
            for line in file:
                sentence = line.strip()
                if not sentence:
                    continue

                if sentence.endswith("<EOS>"):
                    sentence = sentence.removesuffix("<EOS>").strip()

                if sentence:
                    sentences.append(sentence)

        return sentences

    def load_sent_split_text(self, dataset_path):
        with dataset_path.open("r", encoding="utf-8") as file:
            return file.read()

    def build_sentence_mask(self, tokens):
        """
        Create a binary mask over sentence tokens.
        The last token of each sentence is marked as 1.
        """
        return [0] * len(tokens)

    def build_candidate_labels(self, tokens, sentence_mask):
        """
        Keep the current training pipeline on non-letter tokens and newlines.
        """
        candidate_labels = [self.ignore_index] * len(tokens)

        for i, token in enumerate(tokens):
            if token == self.newline_token or not self.is_letter_token(token):
                candidate_labels[i] = sentence_mask[i]

        return candidate_labels

    def tokenize_preserving_newlines(self, text):
        tokens = []

        for chunk in re.split(r"(\n)", text):
            if not chunk:
                continue

            if chunk == "\n":
                tokens.append(self.newline_token)
                continue

            tokens.extend(self.tokenizer.tokenize(chunk))

        return tokens

    def is_letter_token(self, token):
        if token == self.newline_token:
            return False

        normalized_token = token.removeprefix("##")
        return normalized_token.isalpha()

    def mark_pre_eos_non_letter_tokens(self, tokens, sentence_mask):
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            if token == self.newline_token:
                break

            if self.is_letter_token(token):
                break

            sentence_mask[i] = 1

    def encode_text_segment_targets(self, text_segment):
        """
        Tokenize a raw text segment and build both the binary sentence mask and
        pipeline labels.
        """
        tokens = self.tokenize_preserving_newlines(text_segment)
        sentence_mask = self.build_sentence_mask(tokens)
        labels = self.build_candidate_labels(tokens, sentence_mask)
        return tokens, sentence_mask, labels

    def mark_post_eos_newlines(self, text, start_index, corpus_tokens, corpus_sentence_mask):
        current_index = start_index

        while current_index < len(text) and text[current_index] == "\n":
            corpus_tokens.append(self.newline_token)
            corpus_sentence_mask.append(1)
            current_index += 1

        return current_index

    def build_corpus_targets(self):
        """
        Concatenate corpus tokens, full binary sentence mask and pipeline labels
        from the raw .sent_split text.
        """
        dataset_path = Path(self.filename)
        corpus_tokens = []
        corpus_sentence_mask = []

        if dataset_path.suffix == ".sent_split":
            text = self.load_sent_split_text(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        current_index = 0
        eos_marker = "<EOS>"

        while current_index < len(text):
            eos_index = text.find(eos_marker, current_index)
            if eos_index == -1:
                segment = text[current_index:]
                tokens, sentence_mask, _ = self.encode_text_segment_targets(segment)
                corpus_tokens.extend(tokens)
                corpus_sentence_mask.extend(sentence_mask)
                break

            segment = text[current_index:eos_index]
            tokens, sentence_mask, _ = self.encode_text_segment_targets(segment)
            self.mark_pre_eos_non_letter_tokens(tokens, sentence_mask)
            corpus_tokens.extend(tokens)
            corpus_sentence_mask.extend(sentence_mask)
            current_index = eos_index + len(eos_marker)
            current_index = self.mark_post_eos_newlines(
                text,
                current_index,
                corpus_tokens,
                corpus_sentence_mask,
            )

        corpus_labels = self.build_candidate_labels(corpus_tokens, corpus_sentence_mask)
        return corpus_tokens, corpus_sentence_mask, corpus_labels

    def build_vocabulary(self):
        """
        Build token to id mappings for the corpus.
        """
        vocabulary = [self.pad_token, self.unk_token]
        vocabulary.extend(sorted(set(self.corpus_tokens)))

        token_to_id = {token: i for i, token in enumerate(vocabulary)}
        id_to_token = {i: token for token, i in token_to_id.items()}
        return token_to_id, id_to_token

    def encode_corpus_tokens(self):
        """
        Convert corpus tokens into token ids.
        """
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk_id) for token in self.corpus_tokens]

    def build_candidate_positions(self):
        """
        Store all token positions whose labels are valid training targets.
        """
        return [
            i for i, label in enumerate(self.corpus_labels) if label != self.ignore_index
        ]

    def get_candidate_delimiters_as_tokens(self):
        """
        Return all candidate delimiter tokens that actually appear in the corpus.
        """
        delimiters = []

        for token, label in zip(self.corpus_tokens, self.corpus_labels):
            if label != self.ignore_index:
                delimiters.append(token)

        return delimiters

    def is_delimiter_token(self, token):
        """
        Return True if the token is punctuation that can plausibly close a sentence.
        """
        if not token:
            return False

        is_punctuation = all(
            unicodedata.category(char).startswith("P") for char in token
        )
        return is_punctuation and token not in self.excluded_delimiters

    def build_context_window(self, position):
        """
        Build a fixed-size context window centered on a candidate position.
        """
        return self.build_context_window_from_tokens(self.corpus_token_ids, position)

    def build_inference_stream(self):
        """
        Build an inference stream and keep track of line endings.
        """
        dataset_path = Path(self.filename)
        if dataset_path.suffix != ".sent_split":
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        inference_tokens = self.tokenize_preserving_newlines(
            self.load_sent_split_text(dataset_path).replace("<EOS>", "")
        )
        line_end_positions = {
            i for i, token in enumerate(inference_tokens) if token == self.newline_token
        }

        unk_id = self.token_to_id[self.unk_token]
        inference_token_ids = [
            self.token_to_id.get(token, unk_id) for token in inference_tokens
        ]

        return inference_tokens, inference_token_ids, line_end_positions

    def build_inference_context_window(self, token_ids, position):
        """
        Build a fixed-size context window over an arbitrary token stream.
        """
        return self.build_context_window_from_tokens(token_ids, position)

    def build_context_window_from_tokens(self, token_ids, position):
        """
        Build a fixed-size context window over a token-id stream.
        """
        half_window = self.window_size // 2
        left_index = position - half_window
        right_index = position + half_window + 1

        pad_id = self.token_to_id[self.pad_token]
        window_ids = []

        for i in range(left_index, right_index):
            if 0 <= i < len(token_ids):
                window_ids.append(token_ids[i])
            else:
                window_ids.append(pad_id)

        return window_ids

    def __len__(self):
        return len(self.candidate_positions)

    def __getitem__(self, index):
        candidate_position = self.candidate_positions[index]
        window_token_ids = self.build_context_window(candidate_position)
        label = self.corpus_labels[candidate_position]

        return {
            "token_ids": torch.tensor(window_token_ids, dtype=torch.long),
            "candidate_position": torch.tensor(self.window_size // 2, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }
