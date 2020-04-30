import unittest

from nltk.wsd import lesk
from nltk import word_tokenize
from textblob.wordnet import Synset


def get_synset(sentence, word):
    return lesk(word_tokenize(sentence), word)


sentence1 = "Keep your savings in the bank"
sentence2 = "It's so risky to drive over the banks of the road"
print(get_synset(sentence1, 'bank'))
print(get_synset(sentence2, 'bank'))
s = Synset('savings_bank.n.02')



class TestMethods(unittest.TestCase):
    def test_get_lemma(self):
        res = Synset('savings_bank.n.01')
        self.assertEqual(get_synset("Keep your savings in the bank", 'bank'), res)


if __name__ == '__main__':
    unittest.main()
