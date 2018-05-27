import os

from torchtext.data import TabularDataset


class Sentiment140(TabularDataset):
    urls = ['http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip']
    dirname = ''
    name = ''
    train_name = 'training.1600000.processed.noemoticon.csv'
    train_utf8_name = 'training.1600000.processed.noemoticon.utf8.csv'
    test_name = 'testdata.manual.2009.06.14.csv'

    def __init__(self, root, text_field, label_field, train=True, download=False, **kwargs):
        if download:
            root = self.download(root)
        else:
            root = os.path.join(root, self.dirname)

        filename = os.path.join(root, self.train_name)
        utf8_filename = os.path.join(root, self.train_utf8_name)
        if not os.path.isfile(utf8_filename):
            with open(filename, 'rb') as f, open(utf8_filename, 'wb') as fw:
                fw.write(f.read().decode('latin-1').encode('utf-8'))

        fields = [('label', label_field),
                  ('', None),
                  ('', None),
                  ('', None),
                  ('', None),
                  ('text', text_field)]

        super(Sentiment140, self).__init__(os.path.join(root, self.train_utf8_name if train else self.test_name),
                                           'csv', fields, **kwargs)
