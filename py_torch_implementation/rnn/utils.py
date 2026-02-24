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
