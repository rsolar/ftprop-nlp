from collections import OrderedDict

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_vector, num_classes):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_vector.size(1)
        self.hidden_size = 32

        # input: (m, seq_len)
        self.embedding = nn.Sequential(OrderedDict([
            ('embed1', nn.Embedding.from_pretrained(embedding_vector)),
        ]))

        # input: (m, seq_len, embedding_dim)
        self.lstm = nn.Sequential(OrderedDict([
            ('lstm2', nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True)),
        ]))

        # input: (m, hidden_size)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout3', nn.Dropout()),
            ('linear3', nn.Linear(self.hidden_size, num_classes)),
            ('softmax3', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
