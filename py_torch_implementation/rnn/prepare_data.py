import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

from py_torch_implementation.rnn import utils
from py_torch_implementation.rnn.config import Config
from py_torch_implementation.rnn.data_loader import get_data_loader
from py_torch_implementation.rnn.utils import pre_process_dataset


def prepare_data(config: Config) -> tuple[dict[str, DataLoader], int]:
    df = pd.read_csv(config.CSV_PATH).dropna().reset_index(drop=True)
    df['sentiment'] = df['sentiment'].map({"positive": 1., "negative": 0.})

    train_val_data, test_data = train_test_split(
        df.values,
        test_size=config.TEST_SPLIT,
        random_state=42
    )

    train_size = len(train_val_data) - config.VAL_SPLIT
    train_data, valid_data = random_split(train_val_data, [train_size, config.VAL_SPLIT])

    token_labels = utils.get_token_label_matches(train_data)
    vocab_size = len(token_labels)

    dataloaders: dict[str, DataLoader] = {}
    splits = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    for name, subset in splits.items():
        processed_data = pre_process_dataset(subset, token_labels)
        dataloaders[name] = get_data_loader(dataset=processed_data, batch_size=config.BATCH_SIZE)

    return dataloaders, vocab_size
