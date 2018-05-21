from collections import OrderedDict

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, input_shape, num_classes, embedding_vector, nonlin=nn.ReLU):
        super(CNN_LSTM, self).__init__()
        _, self.seq_len, self.embedding_dim = input_shape
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 32
        self.total_num_filters = len(self.filter_sizes) * self.num_filters
        self.hidden_size = 32

        # input: (m, seq_len)
        self.embedding = nn.Sequential(OrderedDict([
            ('embed1', nn.Embedding.from_pretrained(embedding_vector, freeze=False)),
        ]))
        # output: (m, seq_len, embedding_dim)

        # input: (m, 1, seq_len, embedding_dim)
        self.convs = nn.ModuleList()
        for i, filter_size in enumerate(self.filter_sizes):
            self.convs.add_module('conv2' + chr(ord('a') + i),
                                  nn.Sequential(OrderedDict([
                                      ('conv2' + chr(ord('a') + i), nn.Conv2d(1, self.num_filters,
                                                                              (filter_size, self.embedding_dim))),
                                      ('maxpool2' + chr(ord('a') + i), nn.MaxPool2d((self.seq_len - filter_size + 1, 1),
                                                                                    stride=1)),
                                      ('nonlin2' + chr(ord('a') + i), nonlin()),
                                  ])))
        # output: (m, total_num_filters, 1, 1)

        # input: (m, 1, total_num_filters)
        self.lstm = nn.Sequential(OrderedDict([
            ('dropout3', nn.Dropout(0.5)),
            ('lstm3', nn.LSTM(self.total_num_filters, self.hidden_size, batch_first=True)),
        ]))
        # output: (m, 1, hidden_size)

        # input: (m, hidden_size)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc4', nn.Linear(self.hidden_size, num_classes)),
        ]))

    def forward(self, x):
        x = self.embedding(x)

        x = x.unsqueeze(dim=1)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)

        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
