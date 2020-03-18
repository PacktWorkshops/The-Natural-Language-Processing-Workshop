import unittest

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import WordPunctTokenizer


def tokenize_with_tweet_tokenizer(text):
    tweet_tokenizer = TweetTokenizer()  # Here will create an object of tweetTokenizer
    return tweet_tokenizer.tokenize(text)  # Then we will call the tokenize
    # method oftweetTokenizer which will return token list of sentence.


sentence = """Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath,
 New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! 
 @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"""
print(tokenize_with_tweet_tokenizer(sentence))


def tokenize_with_mwe(text):
    mwe_tokenizer = MWETokenizer([('Republic', 'Day')])
    mwe_tokenizer.add_mwe(('Indian', 'Army'))
    return mwe_tokenizer.tokenize(text.split())


print(tokenize_with_mwe(sentence))

print(tokenize_with_mwe(sentence.replace('!', '')))


def tokenize_with_regex_tokenizer(text):
    reg_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    return reg_tokenizer.tokenize(text)


print(tokenize_with_regex_tokenizer(sentence))


def tokenize_with_wst(text):
    wh_tokenizer = WhitespaceTokenizer()
    return wh_tokenizer.tokenize(text)


print(tokenize_with_wst(sentence))


def tokenize_with_wordpunct_tokenizer(text):
    wp_tokenizer = WordPunctTokenizer()
    return wp_tokenizer.tokenize(text)


print(tokenize_with_wordpunct_tokenizer(sentence))


class TestMethods(unittest.TestCase):

    def test_tokenize_with_wordpunct_tokenizer(self):
        tokens = ['For', 'more', 'photos', 'ping', 'me', 'sunil', '@', 'photoking', '.', 'com', ':)']
        self.assertEqual(tokenize_with_wst('Sunil tweeted, "Witnessing 70th Republic Day of India'), tokens)

    def test_tokenize_with_wst(self):
        tokens = ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic', 'Day', 'of', 'India']
        self.assertEqual(tokenize_with_wordpunct_tokenizer(' For more photos ping me sunil@photoking.com :)'), tokens)

    def test_tokenize_with_tweet_tokenizer(self):
        tokens = ['Awesome', 'airshow', '!', '@india_official', '@indian_army', '#India', '#70thRepublic_Day', '.']
        self.assertEqual(
            tokenize_with_tweet_tokenizer('Awesome airshow! @india_official  @indian_army #India #70thRepublic_Day.'),
            tokens)

    def test_tokenize_with_mwe(self):
        tokens = ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic_Day', 'of', 'India']
        self.assertEqual(tokenize_with_mwe('Sunil tweeted, "Witnessing 70th Republic Day of India'), tokens)

    def test_tokenize_with_regex_tokenizer(self):
        tokens = ['Sunil', 'tweeted', ',', '"Witnessing', '70th', 'Republic', 'Day', 'of', 'India']
        self.assertEqual(tokenize_with_regex_tokenizer('Sunil tweeted, "Witnessing 70th Republic Day of India'), tokens)


if __name__ == '__main__':
    unittest.main()
