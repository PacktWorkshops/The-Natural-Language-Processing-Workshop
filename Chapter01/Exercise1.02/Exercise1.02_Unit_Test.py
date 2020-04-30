from nltk import word_tokenize
import unittest


# @author Muzaffar


def get_tokens(sentence):
    words = word_tokenize(sentence)
    return words


print(get_tokens("I am reading NLP Fundamentals."))


class TestMethods(unittest.TestCase):

    def test_get_tokens(self):
        tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

        self.assertEqual(get_tokens('The quick brown fox jumps over the lazy dog'), tokens)


if __name__ == '__main__':
    unittest.main()
