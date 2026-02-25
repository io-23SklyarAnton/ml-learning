import pandas as pd
import torch
from torch import sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

from py_torch_implementation.rnn.data_loader import get_data_loader
from py_torch_implementation.rnn.utils import get_token_label_matches

from py_torch_implementation.rnn.lstm_model import Model as LstmModel

if __name__ == '__main__':
    device = torch.device("mps")

    df = pd.read_csv("../../datasets/IMDB Dataset.csv")

    df = df.dropna().reset_index(drop=True)

    df_copy = df.copy()
    df_copy['sentiment'] = df_copy['sentiment'].replace({"positive": 1., "negative": 0.})

    train_dataset, test_dataset = train_test_split(df_copy.values, test_size=0.1)

    train_dataset, valid_dataset = random_split(train_dataset, [40_000, 5000])

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

    valid_data_loader = get_data_loader(
        dataset=valid_dataset,
        batch_size=100,
        shuffle=False,
        token_labels=token_labels,
    )

    model = LstmModel(
        num_embeddings=len(token_labels),
        embedding_dim=50,
        hidden_size=100
    ).to(device)

    loss_f = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())

    print("LEARNING")
    for epoch in range(3):
        correct_guesses = 0
        total_loss = 0.0
        total_samples = 0

        model.train()
        batches_processed = 0
        for padded_text_list, label_list, lengths in train_data_loader:
            padded_text_list = padded_text_list.to(device)
            label_list = label_list.to(device)
            optimizer.zero_grad()
            predict = model(padded_text_list, lengths)
            loss = loss_f(predict, label_list)
            loss.backward()
            optimizer.step()

            batches_processed += 1
            total_loss += loss.item()
            exact_predict = (predict > 0.0).float()
            correct_guesses += (exact_predict == label_list).sum().item()
            total_samples += label_list.size(0)

            if batches_processed % 100 == 0:
                print(f"processed {batches_processed} batches")

        avg_epoch_loss = total_loss / len(train_data_loader)
        epoch_accuracy = correct_guesses / total_samples
        print(f"epoch {epoch} finished\ntrain loss {avg_epoch_loss}\nepoch_accuracy {epoch_accuracy}\n")

        model.eval()
        correct_guesses = 0
        total_samples = 0
        with torch.no_grad():
            for padded_text_list, label_list, lengths in valid_data_loader:
                padded_text_list = padded_text_list.to(device)
                label_list = label_list.to(device)
                predict = model(padded_text_list, lengths)

                exact_predict = (predict > 0.0).float()
                correct_guesses += (exact_predict == label_list).sum().item()
                total_samples += label_list.size(0)

            epoch_accuracy = correct_guesses / total_samples
            print(f"Validation epoch_accuracy: {epoch_accuracy}\n")

    correct_guesses = 0
    total_samples = 0
    with torch.no_grad():
        for padded_text_list, label_list, lengths in test_data_loader:
            padded_text_list = padded_text_list.to(device)
            label_list = label_list.to(device)
            predict = model(padded_text_list, lengths)
            print(f"predict is {sigmoid(predict[0])}, label is {label_list[0]}")

            exact_predict = (predict > 0.0).float()
            correct_guesses += (exact_predict == label_list).sum().item()
            total_samples += label_list.size(0)

        epoch_accuracy = correct_guesses / total_samples
        print(f"Test accuracy: {epoch_accuracy}\n")
