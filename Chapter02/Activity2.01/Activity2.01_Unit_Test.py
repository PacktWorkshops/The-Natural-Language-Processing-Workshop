# Extracting Top Keywords from the news article
# In this notebook, we will perform the activity of extracting top keywords from news article

import operator
import unittest

from nltk.tokenize import WhitespaceTokenizer
from nltk import download, stem

# The below statement will download the stop word list
# 'nltk_data/corpora/stopwords/' at home directory of your computer
download('stopwords')
from nltk.corpus import stopwords


def load_file(file_path):
    # load the new article
    news = ''.join([line for line in open(file_path, encoding="utf8")])
    return news


def to_lower_case(text):
    return text.lower()


wht = WhitespaceTokenizer()


def tokenize_text(text):
    return wht.tokenize(text=text)


stop_words = stopwords.words('english')


def remove_stop_words(token_list):
    return [word for word in token_list if word not in stop_words]


stemmer = stem.PorterStemmer()


def get_stems(token_list):
    return [stemmer.stem(word) for word in token_list]


def get_freq(stems):
    freq_dict = {}
    for t in stems:
        freq_dict[t.strip()] = freq_dict.get(t.strip(), 0) + 1
    return freq_dict


def get_top_n_words(freq_dict, n):
    sorted_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict][:n]


path = "../data/news_article.txt"
news_article = load_file(path)

lower_case_news_art = to_lower_case(text=news_article)

tokens = tokenize_text(lower_case_news_art)

removed_tokens = remove_stop_words(tokens)
stems = get_stems(removed_tokens)
freq_dict = get_freq(stems)
top_keywords = get_top_n_words(freq_dict, 6)
print(top_keywords)


class TestMethods(unittest.TestCase):

    def test_to_lower_case(self):
        res = 'this is a test string'
        self.assertEqual(to_lower_case("This Is a Test String"), res)

    def test_remove_stop_words(self):
        res = ['test', 'string']
        self.assertEqual(remove_stop_words(["this", "is", "a", "test", "string"]), res)

    def test_get_stems(self):
        res = ['thi', 'string', 'is', 'for', 'test']
        self.assertEqual(get_stems(["this", "string", "is", "for", "testing"]), res)

    def test_get_freq(self):
        res = 2
        self.assertEqual(get_freq(["this", "is", "a", "string", "this",
                                   "is", "for", "testing"])['is'], res)


if __name__ == '__main__':
    unittest.main()
