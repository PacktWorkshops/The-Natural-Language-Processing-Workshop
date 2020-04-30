import re
import unittest


def clean_text(sentence):
    return re.sub(r'([^\s\w]|_)+', ' ', sentence).split()


sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'
print(clean_text(sentence))


class TestMethods(unittest.TestCase):

    def test_clean_text(self):
        tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

        self.assertEqual(clean_text('The quick brown, fox jumps over ., the lazy dog'), tokens)


if __name__ == '__main__':
    unittest.main()
