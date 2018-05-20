from collections import OrderedDict

import torch
import torch.nn as nn


class LSTM_CNN(nn.Module):
    def __init__(self, input_shape, num_classes, embedding_vector, nonlin=nn.ReLU):
        super(LSTM_CNN, self).__init__()
        _, self.seq_len, self.embedding_dim = input_shape
        self.hidden_size = 32
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 32
        self.total_num_filters = len(self.filter_sizes) * self.num_filters

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

        # input: (m, 1, seq_len, hidden_size)
        self.convs = nn.ModuleList()
        for i, filter_size in enumerate(self.filter_sizes):
            self.convs.add_module('conv3' + chr(ord('a') + i),
                                  nn.Sequential(OrderedDict([
                                      ('conv3' + chr(ord('a') + i), nn.Conv2d(1, self.num_filters,
                                                                              (filter_size, self.hidden_size))),
                                      ('nonlin3' + chr(ord('a') + i), nonlin()),
                                      ('maxpool3' + chr(ord('a') + i), nn.MaxPool2d((self.seq_len - filter_size + 1, 1),
                                                                                    stride=1)),
                                  ])))
        # output: (m, total_num_filters, 1, 1)

        # input: (m, total_num_filters)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout4', nn.Dropout(0.5)),
            ('fc4', nn.Linear(self.total_num_filters, num_classes)),
            ('softmax4', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)

        x = x.unsqueeze(dim=1)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
