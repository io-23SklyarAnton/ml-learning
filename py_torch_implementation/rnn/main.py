import pandas as pd
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

from py_torch_implementation.rnn.utils import get_tokens

df = pd.read_csv("../../datasets/IMDB Dataset.csv")

df.dropna()
df.reset_index()

df_copy = df.copy()
df_copy['sentiment'] = df_copy['sentiment'].replace({"positive": 1, "negative": 0})

train_dataset, test_dataset = train_test_split(df_copy.values, test_size=0.5)

train_dataset, valid_dataset = random_split(train_dataset, [20000, 5000])

token_label = get_tokens(train_dataset)

print('Vocab-size:', len(token_label))
