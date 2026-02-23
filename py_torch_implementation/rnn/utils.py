import re


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


def get_token_labels(dataset):
    token_label = {}
    index = 2
    for line, label in dataset:
        tokens = tokenizer(line)

        for token in tokens:
            if token not in token_label:
                token_label[token] = index
                index += 1

    token_label['<pad>'] = 0

    return token_label


def get_token_label(
        token: str,
        token_labels: dict[str, int]
) -> int:
    return token_labels.get(token, 1)
