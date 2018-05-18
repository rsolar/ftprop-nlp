import os

from torchtext.data import TabularDataset


class TSAD(TabularDataset):
    urls = ['http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip']
    dirname = 'Sentiment-Analysis-Dataset'
    filename = 'Sentiment Analysis Dataset_small.csv'
    name = ''

    def __init__(self, root, text_field, label_field, download=False, **kwargs):
        if download:
            self.download(root)

        fields = [('', None), ('label', label_field), ('', None), ('text', text_field)]
        super(TSAD, self).__init__(os.path.join(root, self.dirname, self.filename),
                                   'csv', fields, skip_header=True, **kwargs)
