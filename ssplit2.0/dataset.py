from pathlib import Path

import torch

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:
        pass

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        filename: str,
        window_size: int,
        eos_token: str = "<EOS>",
        space_token: str = "<SPACE>",
        boundary_token: str = "<BND>"
    ) -> None:
        super().__init__()
        self.filename = filename
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.eos_token = eos_token
        self.space_token = space_token
        self.boundary_token = boundary_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [f"{self.space_token}", f"{self.boundary_token}"]})
        self.samples = self._load_sentences()
        self.serialized_corpus = self._serialize_corpus()
        self.corpus, self.target = self._build_corpus_and_target()
        self.dataset = self._create_dataset()

    def _load_sentences(self) -> list[str]:
        text = Path(self.filename).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = " ".join(lines)
        text = " ".join(text.split())

        sentences = [sentence.strip() for sentence in text.split(self.eos_token)]
        return [sentence for sentence in sentences if sentence]

    def _serialize_corpus(self) -> str:
        serialized_sentences = [
            sentence.replace(" ", f" {self.space_token} ")
            for sentence in self.samples
        ]
        corpus = f" {self.boundary_token} ".join(serialized_sentences)
        return " ".join(corpus.split())

    def _tokenize_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        tokens = []

        for word_index, word in enumerate(words):
            if self.tokenizer is None:
                word_tokens = [word]
            else:
                word_tokens = self.tokenizer.tokenize(word)

            tokens.extend(word_tokens)

            if word_index < len(words) - 1:
                tokens.append(self.space_token)

        return tokens

    def _build_corpus_and_target(self) -> tuple[list[str], list[int]]:
        corpus = []
        target = []

        for sentence_index, sentence in enumerate(self.samples):
            sentence_tokens = self._tokenize_sentence(sentence)
            corpus.extend(sentence_tokens)
            target.extend([0] * len(sentence_tokens))

            if sentence_index < len(self.samples) - 1:
                corpus.append(self.space_token)
                target.append(1)

        return corpus, target
    
    def _encode_tokens(self, tokens: list[str]) -> list[int]: 
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def _decode_tokens(self, ids: list[int]) -> list[str]: 
        return self.tokenizer.convert_ids_to_tokens(ids)

    def create_context(self, index: int) -> tuple[list[int], list[int]]:
        start = max(0, index - self.window_size)
        end = min(len(self.corpus), index + self.window_size + 1)
        context_tokens = self.corpus[start:end]

        left_padding = self.window_size - (index - start)
        right_padding = self.window_size - (end - index - 1)

        if left_padding > 0:
            context_tokens = [self.tokenizer.pad_token] * left_padding + context_tokens
        if right_padding > 0:
            context_tokens = context_tokens + [self.tokenizer.pad_token] * right_padding

        context_ids = self._encode_tokens(context_tokens)
        attention_mask = [
            0 if token == self.tokenizer.pad_token else 1
            for token in context_tokens
        ]
        return context_ids, attention_mask

    def _create_dataset(self):
        dataset = []

        for index, token in enumerate(self.corpus):
            if token != self.space_token:
                continue

            input_ids, attention_mask = self.create_context(index)

            dataset.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "label": torch.tensor(self.target[index], dtype=torch.long),
                    "center_index": torch.tensor(index, dtype=torch.long),
                    "marker_position": torch.tensor(self.window_size, dtype=torch.long),
                }
            )

        return dataset
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
