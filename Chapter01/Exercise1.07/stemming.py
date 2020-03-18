from nltk import stem
import unittest


def get_stems(word, stemmer):
    return stemmer.stem(word)


porterStem = stem.PorterStemmer()
get_stems("production", porterStem)
get_stems("coming", porterStem)
get_stems("firing", porterStem)
get_stems("battling", porterStem)

snowball_stemmer = stem.SnowballStemmer("english")
get_stems("battling", snowball_stemmer)


class TestMethods(unittest.TestCase):
    def test_get_stems(self):
        porterStem = stem.PorterStemmer()
        self.assertEqual(get_stems('during', porterStem), 'dure')


if __name__ == '__main__':
    unittest.main()
