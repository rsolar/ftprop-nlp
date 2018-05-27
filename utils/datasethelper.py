from functools import partial

import torch
from torchtext.data import Field

from datasets.semeval import SemEval
from datasets.sentiment140 import Sentiment140
from datasets.tsad import TSAD
from utils.bucketiteratorwrapper import BucketIteratorWrapper
from utils.embedprocessor import initialize_embedding
from utils.textprocessor import tokenize


def create_datasets(ds_name, batch_size, no_val_set, use_cuda, seed):
    ds_name = ds_name.lower()

    if ds_name == 'tsad':
        ds = TSAD
        create_ds_func = partial(create_tsad_dataset, ds=ds, data_dir='data/' + ds_name, download=True)
        num_classes = 2
    elif ds_name == 'sentiment140':
        ds = Sentiment140
        create_ds_func = partial(create_sentiment140_dataset, ds=ds, data_dir='data/' + ds_name, download=True)
        num_classes = 2
    elif ds_name == 'semeval':
        ds = SemEval
        create_ds_func = partial(create_semeval_dataset, ds=ds, data_dir='data/' + ds_name, download=True)
        num_classes = 3
    else:
        raise NotImplementedError("'{}' datasets is not supported".format(ds_name))

    train_loader, val_loader, test_loader, embedding_vector = \
        create_ds_func(batch_size, use_cuda, seed, create_val=not no_val_set)
    return train_loader, val_loader, test_loader, num_classes, embedding_vector


# outdated
def create_tsad_dataset(batch_size, use_cuda, seed, ds=None, val_pct=0.01, test_pct=0.01,
                        data_dir='', download=False, create_val=True, seq_len=60):
    device = -1
    torch.manual_seed(seed)
    if use_cuda:
        device = None
        torch.cuda.manual_seed(seed)

    text_field = Field(sequential=True, use_vocab=True, fix_length=seq_len, tokenize=tokenize, batch_first=True)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True)

    all_ds = ds(data_dir, text_field, label_field, download=download)

    train_ds, test_ds = all_ds.split(split_ratio=1.0 - test_pct, stratified=True, strata_field='label')
    if create_val:
        total_len = len(train_ds)
        train_ds, val_ds = train_ds.split(split_ratio=1.0 - val_pct, stratified=True, strata_field='label')
        assert len(train_ds) + len(val_ds) == total_len
    else:
        val_ds = None

    text_field.build_vocab(train_ds, vectors="glove.6B.50d")
    initialize_embedding(text_field.vocab)
    embedding_vector = text_field.vocab.vectors

    train_loader = BucketIteratorWrapper(train_ds, batch_size, device=device,
                                         train=True, repeat=False, shuffle=True, sort=False)
    val_loader = BucketIteratorWrapper(val_ds, batch_size, device=device,
                                       train=False, repeat=False, shuffle=False, sort=False) if create_val else None
    test_loader = BucketIteratorWrapper(test_ds, batch_size, device=device,
                                        train=False, repeat=False, shuffle=False, sort=False)
    return train_loader, val_loader, test_loader, embedding_vector


def create_sentiment140_dataset(batch_size, use_cuda, seed, ds=None, val_pct=0.01, test_pct=0.01,
                                data_dir='', download=False, create_val=True, seq_len=60):
    device = -1
    torch.manual_seed(seed)
    if use_cuda:
        device = None
        torch.cuda.manual_seed(seed)

    text_field = Field(sequential=True, use_vocab=True, batch_first=True,
                       fix_length=seq_len, tokenize=tokenize)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True,
                        postprocessing=lambda x, y, z: [t / 4 for t in x])

    all_ds = ds(data_dir, text_field, label_field, train=True, download=download)

    if create_val:
        train_ds, val_ds, test_ds = all_ds.split(split_ratio=[1.0 - test_pct - val_pct, test_pct, val_pct],
                                                 stratified=True, strata_field='label')
    else:
        train_ds, test_ds = all_ds.split(split_ratio=1.0 - test_pct, stratified=True, strata_field='label')
        val_ds = None

    text_field.build_vocab(train_ds, vectors="glove.6B.50d")
    initialize_embedding(text_field.vocab)
    embedding_vector = text_field.vocab.vectors

    train_loader = BucketIteratorWrapper(train_ds, batch_size, device=device,
                                         train=True, repeat=False, shuffle=True, sort=False)
    val_loader = BucketIteratorWrapper(val_ds, batch_size, device=device,
                                       train=False, repeat=False, shuffle=False, sort=False) if create_val else None
    test_loader = BucketIteratorWrapper(test_ds, batch_size, device=device,
                                        train=False, repeat=False, shuffle=False, sort=False)
    return train_loader, val_loader, test_loader, embedding_vector


def create_semeval_dataset(batch_size, use_cuda, seed, ds=None, val_pct=0.2,
                           data_dir='', download=False, create_val=True, seq_len=60, tar_len=6):
    device = -1
    torch.manual_seed(seed)
    if use_cuda:
        device = None
        torch.cuda.manual_seed(seed)

    label_dic = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}

    text_field = Field(sequential=True, use_vocab=True, batch_first=True,
                       fix_length=seq_len, tokenize=tokenize)
    target_field = Field(sequential=True, use_vocab=True, batch_first=True,
                         fix_length=tar_len, tokenize=tokenize)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True,
                        preprocessing=lambda x: label_dic[x])

    train_ds = ds(data_dir, text_field, target_field, label_field, train=True, download=download)
    test_ds = ds(data_dir, text_field, target_field, label_field, train=False, download=False)

    if create_val:
        train_ds, val_ds = train_ds.split(split_ratio=1.0 - val_pct)
    else:
        val_ds = None

    text_field.build_vocab(train_ds, vectors="glove.6B.100d")
    target_field.build_vocab(train_ds, vectors="glove.6B.100d")
    initialize_embedding(text_field.vocab)
    initialize_embedding(target_field.vocab)
    embedding_vector = text_field.vocab.vectors, target_field.vocab.vectors

    train_loader = BucketIteratorWrapper(train_ds, batch_size, target=True, device=device,
                                         train=True, repeat=False, shuffle=True, sort=False)
    val_loader = BucketIteratorWrapper(val_ds, batch_size, target=True, device=device,
                                       train=False, repeat=False, shuffle=False, sort=False) if create_val else None
    test_loader = BucketIteratorWrapper(test_ds, batch_size, target=True, device=device,
                                        train=False, repeat=False, shuffle=False, sort=False)
    return train_loader, val_loader, test_loader, embedding_vector
