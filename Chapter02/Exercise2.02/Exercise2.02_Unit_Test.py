import re
import unittest

from nltk import ngrams
from textblob import TextBlob


def n_gram_extractor(sentence, n):
    n_grams = []
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    for i in range(len(tokens) - n + 1):
        n_grams.append(tokens[i:i + n])
    return n_grams


print(n_gram_extractor('The cute little boy is playing with the kitten.', 2))
print(n_gram_extractor('The cute little boy is playing with the kitten.', 3))

print(list(ngrams('The cute little boy is playing with the kitten.'.split(), 2)))
print(list(ngrams('The cute little boy is playing with the kitten.'.split(), 3)))

blob = TextBlob("The cute little boy is playing with the kitten.")
print(blob.ngrams(n=2))

print(blob.ngrams(n=3))


class TestMethods(unittest.TestCase):

    def test_n_gram_extractor(self):
        n_grams = [['The', 'cute'],
                   ['cute', 'little'],
                   ['little', 'boy']]

        self.assertEqual(n_gram_extractor('The cute little boy', 2), n_grams)


if __name__ == '__main__':
    unittest.main()
