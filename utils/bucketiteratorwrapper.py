from torchtext.data.iterator import BucketIterator


class BucketIteratorWrapper(BucketIterator):
    def __init__(self, *args, target=False, **kwargs):
        super(BucketIteratorWrapper, self).__init__(*args, **kwargs)
        self.target = target

    def __iter__(self):
        if self.target:
            for batch in super(BucketIteratorWrapper, self).__iter__():
                yield (batch.text, batch.target), batch.label
        else:
            for batch in super(BucketIteratorWrapper, self).__iter__():
                yield batch.text, batch.label
