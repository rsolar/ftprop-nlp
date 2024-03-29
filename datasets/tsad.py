import os

from torchtext.data import TabularDataset


class TSAD(TabularDataset):
    urls = ['http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip']
    dirname = ''
    name = ''
    # filename = 'Sentiment Analysis Dataset.csv'
    filename = 'Sentiment Analysis Dataset_small.csv'

    def __init__(self, root, text_field, label_field, download=False, **kwargs):
        if download:
            root = self.download(root)
        else:
            root = os.path.join(root, self.dirname)

        fields = [('', None),
                  ('label', label_field),
                  ('', None),
                  ('text', text_field)]

        super(TSAD, self).__init__(os.path.join(root, self.filename),
                                   'csv', fields, skip_header=True, **kwargs)
