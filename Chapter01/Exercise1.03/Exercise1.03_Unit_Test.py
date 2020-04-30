import unittest
from nltk import word_tokenize, pos_tag


def get_tokens(sentence):

    words = word_tokenize(sentence)
    return words


def get_pos(words):
    return pos_tag(words)


class TestMethods(unittest.TestCase):
    def test_get_tokens(self):
        tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        self.assertEqual(get_tokens('The quick brown fox jumps over the lazy dog'), tokens)

    def test_get_pos(self):
        pos_tags = [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'),
                    ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

        self.assertEqual(get_pos(get_tokens('The quick brown fox jumps over the lazy dog')), pos_tags)


if __name__ == '__main__':
    unittest.main()
