import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEOSClassifier(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        hidden_dim=64,
        local_files_only=False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output).squeeze(-1)
        return logits
