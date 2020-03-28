import unittest

from gensim.models import Doc2Vec
import pandas as pd

from IPython.core.display import display

from gensim.parsing.preprocessing import preprocess_string

import warnings

warnings.filterwarnings("ignore")

news_file = '../data/sample_news_data.txt'
with open(news_file, encoding="utf8", errors='ignore') as f:
    news_lines = [line for line in f.readlines()]

lines_df = pd.DataFrame()
indices = list(range(len(news_lines)))
lines_df['news'] = news_lines
lines_df['index'] = indices

display(lines_df.head())

docVecModel = Doc2Vec.load('../data/docVecModel.d2v')


def to_vector(sentence):
    cleaned = preprocess_string(sentence)
    docVector = docVecModel.infer_vector(cleaned)
    return docVector


def similar_news_articles(sentence):
    vector = to_vector(sentence)
    similar_vectors = docVecModel.docvecs.most_similar(positive=[vector])
    similar_lines = lines_df[lines_df.index == similar_vectors[0][0]].news
    return similar_lines


class TestMethods(unittest.TestCase):

    def test_to_vector(self):
        result = -0.0018694705
        # Using delta here because uncertainty in the doc2Vev vectors
        self.assertAlmostEqual(to_vector("US raise TV indecency US politicians are").mean(),
                               result, delta=0.01)


if __name__ == '__main__':
    unittest.main()
