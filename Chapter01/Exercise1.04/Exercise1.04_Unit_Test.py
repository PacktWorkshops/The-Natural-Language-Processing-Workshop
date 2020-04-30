from nltk import download
import unittest

download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)
sentence = "I am learning Python. It is one of the most popular programming languages"
sentence_words = word_tokenize(sentence)
print(sentence_words)


def remove_stop_words(sentence, stop_words):
    return ' '.join([word for word in sentence if word not in stop_words])


print(remove_stop_words(sentence_words, stop_words))
stop_words.extend(['I', 'It', 'one'])
print(remove_stop_words(sentence_words, stop_words))


class TestMethods(unittest.TestCase):
    def test_remove_stop_words(self):
        stop_words = ['The', 'over', 'the']
        tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

        self.assertEqual(remove_stop_words(tokens, stop_words),
                         'quick brown fox jumps lazy dog')


if __name__ == '__main__':
    unittest.main()
