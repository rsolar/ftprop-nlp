from torchtext.data.iterator import BucketIterator


class BucketIteratorWrapper(BucketIterator):
    def __iter__(self):
        for batch in super(BucketIteratorWrapper, self).__iter__():
            yield batch.text, batch.label
