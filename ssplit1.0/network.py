import torch
import torch.nn as nn

class SentenceBoundaryNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_dim,
        num_classes=1,
        padding_idx=0,
        kernel_size=3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.context_encoder = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=context_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.classifier = nn.Linear(context_dim, num_classes)

    def forward(self, token_ids, candidate_positions):
        """
        token_ids: [batch_size, window_size]
        candidate_positions: [batch_size]
        """
        embedded_tokens = self.embedding(token_ids)

        contextualized_tokens = embedded_tokens.transpose(1, 2)
        contextualized_tokens = self.context_encoder(contextualized_tokens)
        contextualized_tokens = torch.relu(contextualized_tokens)
        contextualized_tokens = contextualized_tokens.transpose(1, 2)

        batch_indices = torch.arange(
            candidate_positions.size(0), device=token_ids.device
        )
        candidate_vectors = contextualized_tokens[batch_indices, candidate_positions]

        logits = self.classifier(candidate_vectors)
        return logits.squeeze(-1)
