import unittest

import nltk
from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

nltk.download('wordnet')
sentence = "The products produced by the process today are far better than what it produces generally."
lemmatizer = WordNetLemmatizer()


def get_lemmas(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])


print(get_lemmas(sentence))


class TestMethods(unittest.TestCase):

    def test_get_lemmas(self):
        result = 'why are you going there'
        self.assertEqual(get_lemmas('why are you going there'), result)


if __name__ == '__main__':
    unittest.main()
