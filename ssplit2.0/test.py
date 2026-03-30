from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from network import BertEOSClassifier


WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_FILES = [
    "data/en_merged-ud-dev.sent_split",
    "data/en_merged-ud-test.sent_split",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = BertEOSClassifier(hidden_dim=HIDDEN_DIM)

    # Recreate the tokenizer vocabulary exactly as in training before loading BERT weights.
    _ = TextDataset(model.tokenizer, DEFAULT_FILES[0], WINDOW_SIZE)
    model.bert.resize_token_embeddings(len(model.tokenizer))

    bert_path = CHECKPOINT_DIR / "bert_encoder.pt"
    classifier_path = CHECKPOINT_DIR / "classifier_head.pt"

    model.bert.load_state_dict(torch.load(bert_path, map_location=device))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_file(model, filename):
    dataset = TextDataset(model.tokenizer, filename, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            batch_indices = torch.arange(outputs.size(0), device=outputs.device)
            marker_logits = outputs[batch_indices, batch["marker_position"]]
            positive_logits = marker_logits[:, 1]
            predictions = (torch.sigmoid(positive_logits) >= 0.5).long()

            total_correct += (predictions == batch["label"]).sum().item()
            total_examples += batch["label"].size(0)

    accuracy = total_correct / total_examples
    print(f"{filename} | accuracy={accuracy:.4f} | examples={total_examples}")


def main():
    missing_files = [
        path for path in [
            CHECKPOINT_DIR / "bert_encoder.pt",
            CHECKPOINT_DIR / "classifier_head.pt",
        ]
        if not path.exists()
    ]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Checkpoint mancanti: {missing}")

    model = load_model()

    for filename in DEFAULT_FILES:
        evaluate_file(model, filename)


if __name__ == "__main__":
    main()
