import pandas as pd
import random

SEQUENCE_LENGTH = 128

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    benign = data[data['label'] == 'good']
    malicious = data[data['label'] == 'bad']
    ratio = len(malicious) / len(benign)
    benign = benign.sample(frac=ratio)
    X = pd.concat([benign, malicious])
    return X

def create_vocab():
    vocab = {c: i + 2 for i, c in enumerate(
        '''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-._~:/?#[]@!$&'()*+,;%=''')}
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1
    return vocab

class CharVocab:
    def __init__(self, vocab):
        self.vocab = vocab

    def ctoi(self, c):
        return self.vocab.get(c, self.vocab['[UNK]'])

def preprocess_data(X, vocab, sequence_length=SEQUENCE_LENGTH):
    char_vocab = CharVocab(vocab)
    examples = []
    for _, row in X.iterrows():
        tokens = [char_vocab.ctoi(c) for c in row['URL']][:sequence_length]
        tokens += [char_vocab.ctoi('[PAD]')] * (sequence_length - len(tokens))
        examples.append((tokens, int(row['label'] == 'bad')))
    return examples

def split_data(examples, folds=5):
    random.shuffle(examples)
    ratio = int(1 / folds * len(examples))
    X_folds = []
    for split in range(folds):
        train_set = examples[:split * ratio] + examples[(split + 1) * ratio:]
        test_set = examples[split * ratio:(split + 1) * ratio]
        X_folds.append((train_set, test_set))
    return X_folds