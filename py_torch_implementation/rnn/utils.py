import re

import torch


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(
        pattern='(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        string=text.lower(),
    )
    text = re.sub(
        pattern=r"[^\w\s']",
        repl=' ',
        string=text.lower()
    )
    text += ' '.join(emoticons).replace('-', '')
    return text.split()


def get_token_label_matches(dataset):
    token_label = {}
    index = 2
    for line, label in dataset:
        tokens = tokenizer(line)

        for token in tokens:
            if token not in token_label:
                token_label[token] = index
                index += 1

    token_label['<pad>'] = 0
    token_label['<unk>'] = 1

    return token_label


def get_token_labels(
        tokens: list[str],
        token_labels: dict[str, int]
) -> list[int]:
    return [token_labels.get(token.lower(), 1) for token in tokenizer(tokens)]


def get_device():
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    return torch.device(device)


def pre_process_dataset(raw_dataset, token_labels):
    processed_data = []

    for text, label in raw_dataset:
        token_ids = get_token_labels(text, token_labels)

        text_tensor = torch.tensor(token_ids, dtype=torch.int64)
        processed_data.append((text_tensor, label))

    return processed_data
