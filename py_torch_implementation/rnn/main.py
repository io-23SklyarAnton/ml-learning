import pandas as pd
from torch import sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

from py_torch_implementation.rnn.data_loader import get_data_loader
from py_torch_implementation.rnn.utils import get_token_label_matches

from py_torch_implementation.rnn.model import Model

df = pd.read_csv("../../datasets/IMDB Dataset.csv")

df.dropna()
df.reset_index()

df_copy = df.copy()
df_copy['sentiment'] = df_copy['sentiment'].replace({"positive": 1., "negative": 0.})

train_dataset, test_dataset = train_test_split(df_copy.values, test_size=0.5)

train_dataset, valid_dataset = random_split(train_dataset, [20000, 5000])

token_labels = get_token_label_matches(train_dataset)

train_data_loader = get_data_loader(
    dataset=train_dataset,
    batch_size=30,
    shuffle=True,
    token_labels=token_labels,
)

test_data_loader = get_data_loader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    token_labels=token_labels
)

model = Model(
    num_embeddings=len(token_labels),
    embedding_dim=100,
    hidden_size=100
)

loss_f = BCEWithLogitsLoss()
optimizer = Adam(model.parameters())

for _ in range(5):
    for padded_text_list, label_list, lengths in train_data_loader:
        optimizer.zero_grad()
        predict = model(padded_text_list, lengths)
        loss = loss_f(predict, label_list)
        loss.backward()
        optimizer.step()

    print("epoch finished")

for padded_text_list, label_list, lengths in test_data_loader:
    predict = model(padded_text_list, lengths)
    print(f"predict is {sigmoid(predict[0])}, label is {label_list[0]}")
