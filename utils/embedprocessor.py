import logging

import torch
import torch.nn as nn


def initialize_embedding(vocab, num_special_toks=2):
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = sweep_range - num_special_toks
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            nn.init.normal_(emb_vectors[i], mean=0, std=0.1)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
    logging.info("average GloVE norm is {}, number of known words are {}, total number of words are {}"
                 .format(running_norm / num_non_zero, num_non_zero, total_words))
