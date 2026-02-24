import torch
from torch import nn
from torch.utils.data import DataLoader

from py_torch_implementation.rnn.utils import get_token_labels


def get_data_loader(dataset, token_labels, batch_size, shuffle) -> DataLoader:
    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for _text, _label in batch:
            label_list.append(_label)
            processed_text = torch.tensor(
                data=get_token_labels(_text, token_labels),
                dtype=torch.int64
            )
            text_list.append(processed_text)
            lengths.append(processed_text.size(dim=0))

        label_list = torch.tensor(label_list).reshape(-1, 1)
        lengths = torch.tensor(lengths)
        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

        return padded_text_list, label_list, lengths

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )
