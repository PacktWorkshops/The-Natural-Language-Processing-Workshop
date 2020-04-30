import unittest

from nltk import word_tokenize
from autocorrect import Speller

spell = Speller(lang='en')
spell('Natureal')
sentence = word_tokenize("Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes")
print(sentence)


def correct_spelling(tokens):
    sentence_corrected = ' '.join([spell(word) for word in tokens])
    return sentence_corrected


class TestMethods(unittest.TestCase):
    def test_correct_spelling(self):
        self.assertEqual(correct_spelling('Ntural Luanguage'),
                         'Natural Language')


if __name__ == '__main__':
    unittest.main()
