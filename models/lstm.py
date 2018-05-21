from collections import OrderedDict

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_shape, num_classes, embedding_vector):
        super(LSTM, self).__init__()
        _, _, self.embedding_dim = input_shape
        self.hidden_size = 32

        # input: (m, seq_len)
        self.embedding = nn.Sequential(OrderedDict([
            ('embed1', nn.Embedding.from_pretrained(embedding_vector, freeze=False)),
        ]))
        # output: (m, seq_len, embedding_dim)

        # input: (m, seq_len, embedding_dim)
        self.lstm = nn.Sequential(OrderedDict([
            ('lstm2', nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)),
        ]))
        # output: (m, seq_len, hidden_size)

        # input: (m, hidden_size)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(self.hidden_size, num_classes)),
        ]))

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.classifier(x)
        return x
