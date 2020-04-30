import unittest

import pandas as pd
from IPython.core.display import display
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
import random
import warnings

warnings.filterwarnings("ignore")
sample_news_data = '../data/sample_news_data.txt'

with open(sample_news_data, encoding="utf8", errors='ignore') as f:
    news_lines = [line for line in f.readlines()]
lines_df = pd.DataFrame()

indices = list(range(len(news_lines)))

lines_df['news'] = news_lines
lines_df['index'] = indices

display(lines_df.head())


def preprocess(document):
    return preprocess_string(remove_stopwords(document))


document = lines_df['news'].apply(preprocess)

documents = [TaggedDocument(text, [index])
             for index, text in document.iteritems()]


class DocumentDataset(object):

    def __init__(self, data: pd.DataFrame, column):
        document = data[column].apply(self.preprocess)

        self.documents = [TaggedDocument(text, [index])
                          for index, text in document.iteritems()]

    def preprocess(self, document):
        return preprocess_string(remove_stopwords(document))

    def __iter__(self):
        for document in self.documents:
            yield documents

    def tagged_documents(self, shuffle=False):
        if shuffle:
            random.shuffle(self.documents)
        return self.documents


documents_dataset = DocumentDataset(lines_df, 'news')

docVecModel = Doc2Vec(min_count=1, window=5, vector_size=100, sample=1e-4, negative=5, workers=8)
docVecModel.build_vocab(documents_dataset.tagged_documents())
docVecModel.train(documents_dataset.tagged_documents(shuffle=True),
                  total_examples=docVecModel.corpus_count,
                  epochs=10)
docVecModel.save('../data/docVecModel.d2v')
print(docVecModel[657])


def get_document_vector(document_index):
    return docVecModel[document_index]


import matplotlib.pyplot as plt


def show_image(vector, line):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.tick_params(axis='both',
                   which='both',
                   left=False,
                   bottom=False,
                   top=False,
                   labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    print(line)
    ax.bar(range(len(vector)), vector, 0.5)


def show_news_line(line_number):
    line = lines_df[lines_df.index == line_number].news
    doc_vector = docVecModel[line_number]
    show_image(doc_vector, line)


print(show_news_line(872))


class TestMethods(unittest.TestCase):

    def test_preprocess(self):
        result = ['list', 'success', 'applic', 'call', 'person', 'interview']
        self.assertEqual(preprocess("There is no list, successful applicants will be called for personal interview"),
                         result)


if __name__ == '__main__':
    unittest.main()
