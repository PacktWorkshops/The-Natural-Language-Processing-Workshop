from nltk import download

download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import unittest

lemmatizer = WordNetLemmatizer()


def get_lemma(word):
    return lemmatizer.lemmatize(word)


get_lemma('products')
get_lemma('production')
get_lemma('coming')


class TestMethods(unittest.TestCase):
    def test_get_lemma(self):
        self.assertEqual(get_lemma('during'), 'during')


if __name__ == '__main__':
    unittest.main()
