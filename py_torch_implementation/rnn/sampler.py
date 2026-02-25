import random
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        indices_and_lengths = []
        for idx, (text, label) in enumerate(dataset):
            approximate_length = len(str(text).split())
            indices_and_lengths.append((idx, approximate_length))

        indices_and_lengths.sort(key=lambda x: x[1])

        self.batches = []
        for i in range(0, len(indices_and_lengths), batch_size):
            batch = [idx for idx, _ in indices_and_lengths[i:i + batch_size]]
            self.batches.append(batch)

    def __iter__(self):
        random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
