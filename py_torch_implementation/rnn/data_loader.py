import torch
from torch import nn
from torch.utils.data import DataLoader

from py_torch_implementation.rnn.sampler import BucketBatchSampler


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []

    for text_tensor, label in batch:
        label_list.append(label)
        text_list.append(text_tensor)
        lengths.append(text_tensor.size(0))

    label_list = torch.tensor(label_list, dtype=torch.float32).reshape(-1, 1)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return padded_text_list, label_list, lengths


def get_data_loader(dataset, batch_size) -> DataLoader:
    batch_sampler = BucketBatchSampler(dataset, batch_size)

    return DataLoader(
        dataset=dataset,
        collate_fn=collate_batch,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
