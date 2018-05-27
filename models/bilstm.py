from collections import OrderedDict

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, text_shape, target_shape, num_classes, embedding_vector, nonlin=nn.ReLU):
        super(BiLSTM, self).__init__()
        _, self.seq_len, self.embedding_dim = text_shape
        _, self.tar_len, _ = target_shape
        text_vector, target_vector = embedding_vector
        self.hidden_size = 32
        self.num_layers = 1

        # input: (m, seq_len), (m, tar_len)
        self.embedding_text = nn.Sequential(OrderedDict([
            ('embed1a', nn.Embedding.from_pretrained(text_vector, freeze=False)),
        ]))
        self.embedding_target = nn.Sequential(OrderedDict([
            ('embed1b', nn.Embedding.from_pretrained(target_vector, freeze=False)),
        ]))
        # output: (m, seq_len, embedding_dim), (m, tar_len, embedding_dim)

        # input: (m, seq_len, embedding_dim), (m, tar_len, embedding_dim)
        self.bilstm = nn.Sequential(OrderedDict([
            ('bilstm2', nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers,
                                bidirectional=True, batch_first=True)),
        ]))
        # output: (m, seq_len, hidden_size * 2), (m, tar_len, hidden_size * 2)

        # input: (m, hidden_size * 2, seq_len), (m, hidden_size * 2, tar_len)
        self.pooling_text = nn.Sequential(OrderedDict([
            ('maxpool3a', nn.MaxPool1d(self.seq_len, stride=1)),
        ]))
        self.pooling_target = nn.Sequential(OrderedDict([
            ('maxpool3b', nn.MaxPool1d(self.tar_len, stride=1)),
        ]))
        # output: (m, hidden_size * 2, 1), (m, hidden_size * 2, 1)

        # input: (m, hidden_size * 2, 1), (m, hidden_size * 2, 1)
        self.nonlin = nn.Sequential(OrderedDict([
            ('nonlin4', nonlin()),
        ]))
        # output: (m, hidden_size * 2, 1), (m, hidden_size * 2, 1)

        # input: (m, hidden_size * 4)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout5', nn.Dropout(0.5)),
            ('fc5', nn.Linear(self.hidden_size * 4, num_classes)),
        ]))

    def forward(self, text, target):
        text = self.embedding_text(text)
        target = self.embedding_target(target)

        text, _ = self.bilstm(text)
        target, _ = self.bilstm(target)

        text = text.transpose(1, 2)
        target = target.transpose(1, 2)
        text = self.pooling_text(text)
        target = self.pooling_target(target)

        text = self.nonlin(text)
        target = self.nonlin(target)

        text = text.squeeze(dim=2)
        target = target.squeeze(dim=2)
        x = torch.cat([target, text], dim=1)
        x = self.classifier(x)
        return x
