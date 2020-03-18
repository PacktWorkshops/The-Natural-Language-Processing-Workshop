import unittest

from keras.preprocessing.text import text_to_word_sequence
from textblob import TextBlob, WordList


def get_keras_tokens(text):
    return text_to_word_sequence(text)


sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performancesby Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'

print(get_keras_tokens(sentence))


def get_textblob_tokens(text):
    blob = TextBlob(text)
    return blob.words


print(get_textblob_tokens(sentence))


class TestMethods(unittest.TestCase):

    def test_get_textblob_tokens(self):
        tokens = ['This', 'is', 'a', 'cat']
        self.assertEqual(get_textblob_tokens('This is a cat'), tokens)

    def test_get_keras_tokens(self):
        wrd = WordList(['This', 'is', 'a', 'cat'])
        self.assertEqual(get_keras_tokens('This is a cat'), wrd)


if __name__ == '__main__':
    unittest.main()
