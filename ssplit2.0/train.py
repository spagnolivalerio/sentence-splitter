from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from dataset import TextDataset
from network import BertEOSClassifier
from sampler import BalancedBatchSampler

NUM_EPOCHS = 5
BERT_LR = 2e-5
CLASSIFIER_LR = 1e-4
BATCH_SIZE = 64
POSITIVES_PER_BATCH = 8
WINDOW_SIZE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertEOSClassifier()
tokenizer = model.tokenizer
optimizer = AdamW(
    [
        {"params": model.bert.parameters(), "lr": BERT_LR},
        {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
    ]
)
criterion = nn.BCEWithLogitsLoss()

trainDataset = TextDataset(tokenizer, "data/en_merged-ud-train.sent_split", WINDOW_SIZE)
devDataset = TextDataset(tokenizer, "data/en_merged-ud-dev.sent_split", WINDOW_SIZE)
testDataset = TextDataset(tokenizer, "data/en_merged-ud-test.sent_split", WINDOW_SIZE)

model.bert.resize_token_embeddings(len(tokenizer))
model.to(device)

trainBatchSampler = BalancedBatchSampler(
    trainDataset,
    batch_size=BATCH_SIZE,
    positives_per_batch=POSITIVES_PER_BATCH,
)
trainDataloader = DataLoader(trainDataset, batch_sampler=trainBatchSampler)
devDataloader = DataLoader(devDataset, batch_size=BATCH_SIZE, shuffle=False)
testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

total_training_steps = NUM_EPOCHS * len(trainDataloader)
num_warmup_steps = total_training_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_training_steps,
)

print(
    f"Train sampler | positives={len(trainBatchSampler.positive_indices)} | "
    f"negatives={len(trainBatchSampler.negative_indices)} | "
    f"batch_size={BATCH_SIZE} | positives_per_batch>={POSITIVES_PER_BATCH}"
)
print(f"Train sampler | batches_per_epoch={len(trainBatchSampler)}")


def move_batch_to_device(batch):
    return {key: value.to(device) for key, value in batch.items()}


def compute_batch_loss_and_predictions(batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    batch_indices = torch.arange(outputs.size(0), device=outputs.device)
    positive_logits = outputs[batch_indices, batch["marker_position"]]
    labels = batch["label"].float()

    loss = criterion(positive_logits, labels)
    predictions = (torch.sigmoid(positive_logits) >= 0.5).long()
    return loss, predictions


def compute_binary_counts(predictions, labels):
    predictions = predictions.long()
    labels = labels.long()

    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
    correct = (predictions == labels).sum().item()
    total = labels.numel()

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "correct": correct,
        "total": total,
    }


def metrics_from_counts(counts):
    precision = counts["true_positives"] / max(
        counts["true_positives"] + counts["false_positives"],
        1,
    )
    recall = counts["true_positives"] / max(
        counts["true_positives"] + counts["false_negatives"],
        1,
    )
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = counts["correct"] / max(counts["total"], 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(dataloader):
    model.eval()
    total_loss = 0.0
    total_counts = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "correct": 0,
        "total": 0,
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch)
            loss, predictions = compute_batch_loss_and_predictions(batch)
            batch_counts = compute_binary_counts(predictions, batch["label"])

            total_loss += loss.item()
            for key, value in batch_counts.items():
                total_counts[key] += value

    average_loss = total_loss / len(dataloader)
    metrics = metrics_from_counts(total_counts)
    return average_loss, metrics


for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    total_train_counts = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "correct": 0,
        "total": 0,
    }

    for batch_index, batch in enumerate(trainDataloader, start=1):
        batch = move_batch_to_device(batch)

        optimizer.zero_grad()
        loss, predictions = compute_batch_loss_and_predictions(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_counts = compute_binary_counts(predictions, batch["label"])
        batch_metrics = metrics_from_counts(batch_counts)
        total_train_loss += loss.item()
        for key, value in batch_counts.items():
            total_train_counts[key] += value

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Batch {batch_index}/{len(trainDataloader)} | "
            f"loss={loss.item():.4f} | "
            f"acc={batch_metrics['accuracy']:.4f} | "
            f"f1={batch_metrics['f1']:.4f}"
        )

    train_loss = total_train_loss / len(trainDataloader)
    train_metrics = metrics_from_counts(total_train_counts)
    dev_loss, dev_metrics = evaluate(devDataloader)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_metrics['accuracy']:.4f} | "
        f"train_f1={train_metrics['f1']:.4f} | "
        f"dev_loss={dev_loss:.4f} | "
        f"dev_acc={dev_metrics['accuracy']:.4f} | "
        f"dev_f1={dev_metrics['f1']:.4f}"
    )


test_loss, test_metrics = evaluate(testDataloader)
print(
    f"Test | loss={test_loss:.4f} | "
    f"acc={test_metrics['accuracy']:.4f} | "
    f"f1={test_metrics['f1']:.4f}"
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

torch.save(model.bert.state_dict(), checkpoint_dir / "bert_encoder.pt")
torch.save(model.classifier.state_dict(), checkpoint_dir / "classifier_head.pt")
