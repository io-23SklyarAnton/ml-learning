import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split

from py_torch_implementation.rnn.utils import get_token_labels, get_token_label

df = pd.read_csv("../../datasets/IMDB Dataset.csv")

df.dropna()
df.reset_index()

df_copy = df.copy()
df_copy['sentiment'] = df_copy['sentiment'].replace({"positive": 1., "negative": 0.})

train_dataset, test_dataset = train_test_split(df_copy.values, test_size=0.5)

train_dataset, valid_dataset = random_split(train_dataset, [20000, 5000])

token_label = get_token_labels(train_dataset)

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(_label)
        processed_text = torch.tensor(
            data=get_token_label(_text, token_label),
            dtype=torch.int64
        )
        text_list.append(processed_text)
        lengths.append(processed_text.size(dim=0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return padded_text_list, label_list, lengths


dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False, collate_fn=collate_batch)
