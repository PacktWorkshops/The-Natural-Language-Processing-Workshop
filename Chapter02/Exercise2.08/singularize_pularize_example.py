import unittest

from textblob import TextBlob

sentence = TextBlob('She sells seashells on the seashore')
print(sentence.words)


def singularize(word):
    return word.singularize()


print(singularize(sentence.words[2]))


def pillularize(word):
    return word.pluralize()


print(pillularize(sentence.words[5]))


class TestMethods(unittest.TestCase):

    def test_singularize(self):
        result = 'seashell'
        self.assertEqual(singularize(sentence.words[2]), result)

    def test_pillularize(self):
        result = 'seashores'
        self.assertEqual(pillularize(sentence.words[5]), result)


if __name__ == '__main__':
    unittest.main()
