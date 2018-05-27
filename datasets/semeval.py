import os

from torchtext.data import TabularDataset


class SemEval(TabularDataset):
    urls = ['http://alt.qcri.org/semeval2016/task6/data/uploads/stancedataset.zip']
    dirname = 'StanceDataset'
    name = ''
    train_name = 'train.csv'
    train_utf8_name = 'train.utf8.csv'
    test_name = 'test.csv'

    def __init__(self, root, text_field, target_field, label_field, train=True, download=False, **kwargs):
        if download:
            root = self.download(root)
        else:
            root = os.path.join(root, self.dirname)

        filename = os.path.join(root, self.train_name)
        utf8_filename = os.path.join(root, self.train_utf8_name)
        if not os.path.isfile(utf8_filename):
            with open(filename, 'rb') as f, open(utf8_filename, 'wb') as fw:
                fw.write(f.read().decode('latin-1').encode('utf-8'))

        fields = [('text', text_field),
                  ('target', target_field),
                  ('label', label_field),
                  ('', None),
                  ('', None)]

        super(SemEval, self).__init__(os.path.join(root, self.train_utf8_name if train else self.test_name),
                                      'csv', fields, skip_header=True, **kwargs)
