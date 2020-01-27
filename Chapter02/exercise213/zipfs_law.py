from pylab import *
import nltk

nltk.download('stopwords')
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re
import string
from collections import Counter


def get_stop_words():
    stop_words = stopwords.words('english')
    stop_words = stop_words + list(string.printable)
    return stop_words


def get_and_prepare_data(stop_words):
    """
    This method will load 20newsgroups data and
    and remove stop words from it using given stop word list.
    :param stop_words:
    :return:
    """
    newsgroups_data_sample = fetch_20newsgroups(subset='train')
    tokenized_corpus = [word.lower() for sentence in newsgroups_data_sample['data'] \
                        for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', sentence)) \
                        if word.lower() not in stop_words]
    return tokenized_corpus


def get_frequency(corpus, n):
    token_count_di = Counter(corpus)
    return token_count_di.most_common(n)


def get_actual_and_expected_frequencies(corpus):
    freq_dict = get_frequency(corpus, 1000)
    actual_frequencies = []
    expected_frequencies = []
    for rank, tup in enumerate(freq_dict):
        actual_frequencies.append(log(tup[1]))
        rank = 1 if rank == 0 else rank
        expected_frequencies.append(1 / rank)
    return actual_frequencies, expected_frequencies


def plot(actual_frequencies, expected_frequencies):
    plt.plot(actual_frequencies, 'g*', expected_frequencies, 'ro')
    plt.show()


if __name__ == '__main__':
    stop_word_list = get_stop_words()
    corpus = get_and_prepare_data(stop_word_list)
    print(get_frequency(corpus, 50))
    actual_frequencies, expected_frequencies = get_actual_and_expected_frequencies(corpus)
    plot(actual_frequencies, expected_frequencies)
