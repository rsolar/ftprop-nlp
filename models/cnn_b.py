from collections import OrderedDict

import torch.nn as nn


class CNN_B(nn.Module):
    def __init__(self, input_shape, num_classes, embedding_vector, nonlin=nn.ReLU):
        super(CNN_B, self).__init__()
        _, self.seq_len, self.embedding_dim = input_shape
        self.num_filters = [600, 300, 150, 75]
        self.kernel_size = 3

        # input: (m, seq_len)
        self.embedding = nn.Sequential(OrderedDict([
            ('embed1', nn.Embedding.from_pretrained(embedding_vector)),
            ('dropout1', nn.Dropout(0.4)),
        ]))
        # output: (m, seq_len, embedding_dim)

        # input: (m, embedding_dim, seq_len)
        self.convs = nn.Sequential(OrderedDict([
            ('conv2a', nn.Conv1d(self.embedding_dim, self.num_filters[0], self.kernel_size)),
            ('nonlin2a', nonlin()),
            ('conv2b', nn.Conv1d(self.num_filters[0], self.num_filters[1], self.kernel_size)),
            ('nonlin2b', nonlin()),
            ('conv2c', nn.Conv1d(self.num_filters[1], self.num_filters[2], self.kernel_size)),
            ('nonlin2c', nonlin()),
            ('conv2d', nn.Conv1d(self.num_filters[2], self.num_filters[3], self.kernel_size)),
            ('nonlin2d', nonlin()),
        ]))
        # output: (m, new_dim, num_filter)

        # input: (m, num_filter * new_dim)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc3a', nn.Linear(self.num_filters[3] * (self.seq_len - len(self.num_filters) * (self.kernel_size - 1)),
                               600)),
            ('dropout3', nn.Dropout(0.5)),
            ('nonlin3', nonlin()),
            ('fc3b', nn.Linear(600, num_classes)),
            ('softmax3', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        x = self.embedding(x)
        x.transpose_(1, 2)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
