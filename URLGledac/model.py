import torch
import torch.nn as nn
import torch.nn.functional as F

class URLGledac(nn.Module):
    def __init__(self, vocab_size, sequence_length=128, embedding_size=32, hidden_size=128):
        super(URLGledac, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=2, dropout=0.2)
        self.pool = nn.MaxPool1d(sequence_length)
        self.dense = nn.Linear(hidden_size * 2, 128)
        self.batch = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(128, 2)

    def forward(self, url):
        embedding = self.embedding(url)
        out, _ = self.lstm(torch.transpose(embedding, 0, 1))
        out = self.pool(torch.transpose(out, 0, 2))
        out = self.dense(torch.transpose(out.squeeze(dim=2), 0, 1))
        out = F.relu(out)
        out = self.batch(out)
        out = self.output(out)
        scores = F.log_softmax(out, dim=1)
        return scores