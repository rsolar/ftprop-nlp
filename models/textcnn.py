from collections import OrderedDict

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, input_shape, num_classes, embedding_vector, nonlin=nn.ReLU):
        super(TextCNN, self).__init__()
        _, self.seq_len, self.embedding_dim = input_shape
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.total_num_filters = self.num_filters * len(self.filter_sizes)

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

        # input: (m, total_num_filters)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(self.total_num_filters, num_classes)),
        ]))

    def forward(self, x):
        x = self.embedding(x)

        x = x.unsqueeze(dim=1)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
