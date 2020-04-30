import unittest

from nltk import word_tokenize

sentence = "She sells seashells on the seashore"


def remove_stop_words(text, stop_word_list):
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_word_list])


custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']
print(remove_stop_words(sentence, custom_stop_word_list))


class TestMethods(unittest.TestCase):

    def test_remove_stop_words(self):
        result = 'I am doing'
        self.assertEqual(remove_stop_words('I am doing it', ['it']), result)


if __name__ == '__main__':
    unittest.main()
