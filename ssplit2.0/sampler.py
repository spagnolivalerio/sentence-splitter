import math
import random

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, positives_per_batch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positives_per_batch = max(1, min(positives_per_batch, batch_size))
        self.negatives_per_batch = self.batch_size - self.positives_per_batch

        self.positive_indices = []
        self.negative_indices = []

        for index, sample in enumerate(self.dataset):
            if int(sample["label"]) == 1:
                self.positive_indices.append(index)
            else:
                self.negative_indices.append(index)

        if not self.positive_indices:
            raise ValueError("BalancedBatchSampler requires at least one positive sample.")

        if not self.negative_indices:
            raise ValueError("BalancedBatchSampler requires at least one negative sample.")

    def __len__(self):
        positive_batches = math.ceil(
            len(self.positive_indices) / self.positives_per_batch
        )

        if self.negatives_per_batch == 0:
            return positive_batches

        negative_batches = math.ceil(
            len(self.negative_indices) / self.negatives_per_batch
        )
        return max(positive_batches, negative_batches)

    def _take_from_pool(self, pool, cursor, amount):
        selected = []

        while len(selected) < amount:
            if cursor >= len(pool):
                random.shuffle(pool)
                cursor = 0

            take = min(amount - len(selected), len(pool) - cursor)
            selected.extend(pool[cursor : cursor + take])
            cursor += take

        return selected, cursor

    def __iter__(self):
        positive_pool = list(self.positive_indices)
        negative_pool = list(self.negative_indices)
        random.shuffle(positive_pool)
        random.shuffle(negative_pool)

        positive_cursor = 0
        negative_cursor = 0

        for _ in range(len(self)):
            negative_count = min(
                self.negatives_per_batch,
                len(negative_pool) - negative_cursor,
            )
            positive_count = self.batch_size - negative_count

            batch_indices, positive_cursor = self._take_from_pool(
                positive_pool,
                positive_cursor,
                positive_count,
            )

            if negative_count > 0:
                batch_indices.extend(
                    negative_pool[negative_cursor : negative_cursor + negative_count]
                )
                negative_cursor += negative_count

            random.shuffle(batch_indices)
            yield batch_indices
