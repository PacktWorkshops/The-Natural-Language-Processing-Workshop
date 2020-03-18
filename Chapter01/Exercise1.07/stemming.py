from nltk import stem
import unittest


def get_stems(word, stemmer):
    return stemmer.stem(word)


porterStem = stem.PorterStemmer()
print(get_stems("production", porterStem))
print(get_stems("coming", porterStem))
print(get_stems("firing", porterStem))
print(get_stems("battling", porterStem))

snowball_stemmer = stem.SnowballStemmer("english")
print(get_stems("battling", snowball_stemmer))


class TestMethods(unittest.TestCase):
    def test_get_stems(self):
        porterStem = stem.PorterStemmer()
        self.assertEqual(get_stems('during', porterStem), 'dure')


if __name__ == '__main__':
    unittest.main()
