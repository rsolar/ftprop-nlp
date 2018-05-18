from collections import OrderedDict

import torch
import torch.nn as nn


class LSTM_CNN(nn.Module):
    def __init__(self, embedding_vector, num_classes, nonlin=nn.ReLU):
        super(LSTM_CNN, self).__init__()
        self.embedding_dim = embedding_vector.size(1)
        self.hidden_size = 32
        self.filter_sizes = [3]
        self.num_filters = 32
        self.total_num_filters = len(self.filter_sizes) * self.num_filters

        # input: (m, seq_len)
        self.embedding = nn.Sequential(OrderedDict([
            ('embed1', nn.Embedding.from_pretrained(embedding_vector)),
        ]))

        # input: (m, seq_len, embedding_dim)
        self.lstm = nn.Sequential(OrderedDict([
            ('lstm2', nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True)),
        ]))

        # input: (m, 1, seq_len, hidden_size)
        self.convs = [nn.Sequential(OrderedDict([
            ('conv3' + chr(ord('a') + i), nn.Conv2d(1, self.num_filters, (kernel_size, 1))),
            ('nonlin3' + chr(ord('a') + i), nonlin()),
            ('maxpool3' + chr(ord('a') + i), nn.MaxPool2d((8 - kernel_size, 1), stride=1)),
        ])) for i, kernel_size in enumerate(self.filter_sizes)]
        # output: (m, total_num_filters, seq_len`, hidden_size)

        # input: (m, total_num_filters * hidden_size)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout4', nn.Dropout()),
            ('linear4', nn.Linear(self.total_num_filters * self.hidden_size, num_classes)),
            ('softmax4', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x.unsqueeze_(dim=1)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = x[:, :, -1, :].contiguous()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
