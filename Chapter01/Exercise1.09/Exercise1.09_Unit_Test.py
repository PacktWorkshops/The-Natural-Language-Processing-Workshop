import unittest

from nltk import download, Tree
from nltk import pos_tag
from nltk import ne_chunk
from nltk import word_tokenize

download('maxent_ne_chunker')
download('words')


def get_ner(text):
    i = ne_chunk(pos_tag(word_tokenize(text)), binary=True)
    return [a for a in i if len(a) == 1]


sentence = "India is the second most populous country"
print(get_ner(sentence))


class TestMethods(unittest.TestCase):
    def test_get_lemma(self):
        res = [Tree('NE', [('India', 'NNP')])]
        self.assertEqual(get_ner("India is the second most populous country"), res)


if __name__ == '__main__':
    unittest.main()
