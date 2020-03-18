import unittest

from nltk.stem.porter import *

sentence = "Before eating, it would be nice to sanitize your hands with a sanitizer"

# It is better to create object of PorterStemmer here. 
#  out side method
ps_stemmer = PorterStemmer()


def get_stems(text):
    return ' '.join([ps_stemmer.stem(wd) for wd in text.split()])


print(get_stems(sentence))


class TestMethods(unittest.TestCase):

    def test_get_stems(self):
        res = 'whi are you do this.'
        self.assertEqual(get_stems('Why are you doing this.'), res)


if __name__ == '__main__':
    unittest.main()
