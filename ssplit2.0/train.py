from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from dataset import TextDataset
from network import BertEOSClassifier

NUM_EPOCHS = 10
BERT_LR = 2e-5
CLASSIFIER_LR = 1e-4
BATCH_SIZE = 64
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

trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
devDataloader = DataLoader(devDataset, batch_size=BATCH_SIZE, shuffle=False)
testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

total_training_steps = NUM_EPOCHS * len(trainDataloader)
num_warmup_steps = total_training_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_training_steps,
)


def move_batch_to_device(batch):
    return {key: value.to(device) for key, value in batch.items()}


def compute_batch_loss_and_predictions(batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    batch_size = outputs.size(0)
    batch_indices = torch.arange(batch_size, device=outputs.device)
    marker_logits = outputs[batch_indices, batch["marker_position"]]
    positive_logits = marker_logits[:, 1]
    labels = batch["label"].float()

    loss = criterion(positive_logits, labels)
    predictions = (torch.sigmoid(positive_logits) >= 0.5).long()
    return loss, predictions


def evaluate(dataloader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch)
            loss, predictions = compute_batch_loss_and_predictions(batch)

            total_loss += loss.item()
            total_correct += (predictions == batch["label"]).sum().item()
            total_examples += batch["label"].size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return average_loss, accuracy


for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_examples = 0

    for batch_index, batch in enumerate(trainDataloader, start=1):
        batch = move_batch_to_device(batch)

        optimizer.zero_grad()
        loss, predictions = compute_batch_loss_and_predictions(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        total_train_correct += (predictions == batch["label"]).sum().item()
        total_train_examples += batch["label"].size(0)

        batch_accuracy = (predictions == batch["label"]).float().mean().item()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Batch {batch_index}/{len(trainDataloader)} | "
            f"loss={loss.item():.4f} | acc={batch_accuracy:.4f}"
        )

    train_loss = total_train_loss / len(trainDataloader)
    train_accuracy = total_train_correct / total_train_examples
    dev_loss, dev_accuracy = evaluate(devDataloader)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
        f"dev_loss={dev_loss:.4f} | dev_acc={dev_accuracy:.4f}"
    )


test_loss, test_accuracy = evaluate(testDataloader)
print(f"Test | loss={test_loss:.4f} | acc={test_accuracy:.4f}")

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

torch.save(model.bert.state_dict(), checkpoint_dir / "bert_encoder.pt")
torch.save(model.classifier.state_dict(), checkpoint_dir / "classifier_head.pt")
