from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CorpusDataset
from network import SentenceBoundaryNetwork

TRAIN_FILE_NAME = "sent_split_data/EN_Merged_dataset/en_merged-ud-train.sent_split"
DEV_FILE_NAME = "sent_split_data/EN_Merged_dataset/en_merged-ud-dev.sent_split"
TEST_FILE_NAME = "sent_split_data/EN_Merged_dataset/en_merged-ud-test.sent_split"
BATCH_SIZE = 32
WINDOW_SIZE = 21
EMBEDDING_DIM = 64
CONTEXT_DIM = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
OUTPUT_DIR = Path("inference_logs")


def build_loss(dataset, device):
    positive_count = sum(
        dataset.corpus_labels[position] == 1 for position in dataset.candidate_positions
    )
    negative_count = sum(
        dataset.corpus_labels[position] == 0 for position in dataset.candidate_positions
    )

    if positive_count == 0:
        return nn.BCEWithLogitsLoss()

    pos_weight = torch.tensor([negative_count / positive_count], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    true_positives = 0
    false_positives = 0

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            candidate_positions = batch["candidate_position"].to(device)
            labels = batch["label"].to(device)

            logits = model(token_ids, candidate_positions)
            loss = loss_fn(logits, labels)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            total_correct += (predictions == labels).sum().item()
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return average_loss, accuracy, precision


def decode_tokens(tokens):
    sentence = ""

    for token in tokens:
        if token in {"<PAD>", "<UNK>", "\n"}:
            continue

        if token.startswith("##"):
            sentence += token[2:]
        elif not sentence:
            sentence = token
        elif all(not char.isalnum() for char in token):
            sentence += token
        else:
            sentence += " " + token

    return sentence.strip()


def split_test_corpus(model, dataset, device):
    model.eval()
    predicted_positive_positions = set()
    inference_tokens, inference_token_ids, line_end_positions = dataset.build_inference_stream()

    with torch.no_grad():
        for position, token in enumerate(inference_tokens):
            if position in line_end_positions:
                if dataset.is_delimiter_token(token):
                    predicted_positive_positions.add(position)
                    continue

            should_classify = dataset.is_delimiter_token(token) or position in line_end_positions
            if not should_classify:
                continue

            window_token_ids = dataset.build_inference_context_window(
                inference_token_ids, position
            )
            token_ids = torch.tensor(window_token_ids, dtype=torch.long).unsqueeze(0).to(
                device
            )
            candidate_positions = torch.tensor(
                [dataset.window_size // 2], dtype=torch.long
            ).to(device)

            logits = model(token_ids, candidate_positions)
            probability = torch.sigmoid(logits).item()

            if probability >= 0.5:
                predicted_positive_positions.add(position)

    predicted_sentences = []
    current_sentence = []

    for position, token in enumerate(inference_tokens):
        current_sentence.append(token)

        if position in predicted_positive_positions:
            predicted_sentences.append(decode_tokens(current_sentence))
            current_sentence = []

    if current_sentence:
        predicted_sentences.append(decode_tokens(current_sentence))

    return predicted_sentences


def write_split_outputs(predicted_sentences, dataset):
    OUTPUT_DIR.mkdir(exist_ok=True)

    ground_truth_sentences = dataset.load_sent_split_sentences(Path(dataset.filename))
    comparison_file = OUTPUT_DIR / "test_split_comparison.log"
    predicted_file = OUTPUT_DIR / "test_split_predicted.sent_split"

    with comparison_file.open("w", encoding="utf-8") as file:
        for index, (predicted, ground_truth) in enumerate(
            zip_longest(predicted_sentences, ground_truth_sentences, fillvalue=""),
            start=1,
        ):
            file.write(f"Sample {index}\n")
            file.write(f"predicted: {predicted}\n")
            file.write(f"ground_truth: {ground_truth}\n\n")

    with predicted_file.open("w", encoding="utf-8") as file:
        for sentence in predicted_sentences:
            file.write(f"{sentence}<EOS>\n")

    return comparison_file, predicted_file


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    train_dataset = CorpusDataset(TRAIN_FILE_NAME, tokenizer, window_size=WINDOW_SIZE)
    dev_dataset = CorpusDataset(
        DEV_FILE_NAME,
        tokenizer,
        window_size=WINDOW_SIZE,
        token_to_id=train_dataset.token_to_id,
    )
    test_dataset = CorpusDataset(
        TEST_FILE_NAME,
        tokenizer,
        window_size=WINDOW_SIZE,
        token_to_id=train_dataset.token_to_id,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SentenceBoundaryNetwork(
        vocab_size=len(train_dataset.token_to_id),
        embedding_dim=EMBEDDING_DIM,
        context_dim=CONTEXT_DIM,
        padding_idx=train_dataset.token_to_id[train_dataset.pad_token],
    ).to(device)

    loss_fn = build_loss(train_dataset, device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Train file: {train_dataset.filename}")
    print(f"Validation file: {dev_dataset.filename}")
    print(f"Test file: {test_dataset.filename}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(dev_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0

        for batch in train_dataloader:
            token_ids = batch["token_ids"].to(device)
            candidate_positions = batch["candidate_position"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(token_ids, candidate_positions)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            total_correct += (predictions == labels).sum().item()

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples
        dev_loss, dev_accuracy, dev_precision = evaluate(
            model, dev_dataloader, loss_fn, device
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- train_accuracy: {train_accuracy:.4f} "
            f"- dev_loss: {dev_loss:.4f} "
            f"- dev_accuracy: {dev_accuracy:.4f} "
            f"- dev_precision: {dev_precision:.4f}"
        )

    test_loss, test_accuracy, test_precision = evaluate(
        model, test_dataloader, loss_fn, device
    )
    print(
        f"\nTest metrics - loss: {test_loss:.4f} "
        f"- accuracy: {test_accuracy:.4f} "
        f"- precision: {test_precision:.4f}"
    )

    predicted_sentences = split_test_corpus(model, test_dataset, device)
    comparison_file, predicted_file = write_split_outputs(
        predicted_sentences, test_dataset
    )
    print(f"Comparison log: {comparison_file}")
    print(f"Predicted split file: {predicted_file}")


if __name__ == "__main__":
    main()
