import unittest

from nltk.tokenize import sent_tokenize


def get_sentences(text):
    return sent_tokenize(text)


print(get_sentences(
    "We are reading a book. Do you know who is the publisher? It is Packt. Packt is based out of Birmingham."))
print(
    get_sentences("Mr. Donald John Trump is current president of USA. Before joining politics, he was a businessman."))


class TestMethods(unittest.TestCase):
    def test_get_sentences(self):
        self.assertEqual(get_sentences("Hello, all readers. This is just test string"),
                         ['Hello, all readers.', 'This is just test string'])


if __name__ == '__main__':
    unittest.main()
