#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[26]:


def get_stop_words():
    """
    >>> list(get_stop_words()[:7])
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours']
    """
    stop_words = stopwords.words('english')
    stop_words = stop_words + list(string.printable)
    return stop_words


# In[3]:


def get_and_prepare_data(stop_words):
    """
    This method will load 20newsgroups data and 
    and remove stop words from it using given stop word list.
    :param stop_words: 
    :return: 
    """
    newsgroups_data_sample = fetch_20newsgroups(subset='train')
    tokenized_corpus = [word.lower() for sentence in newsgroups_data_sample['data']                         for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', sentence))                         if word.lower() not in stop_words]
    return tokenized_corpus


# In[35]:


def get_frequency(corpus, n):
    """
    >>> get_frequency(['this', 'is','a','cat'],4)
    [('is', 1), ('a', 1), ('cat', 1), ('this', 1)]
    
    """
    token_count_di = Counter(corpus)
    return token_count_di.most_common(n)


# In[5]:


stop_word_list = get_stop_words()
corpus = get_and_prepare_data(stop_word_list)
get_frequency(corpus, 50)


# In[6]:


def get_actual_and_expected_frequencies(corpus):
    freq_dict = get_frequency(corpus, 1000)
    actual_frequencies = []
    expected_frequencies = []
    for rank, tup in enumerate(freq_dict):
        actual_frequencies.append(log(tup[1]))
        rank = 1 if rank == 0 else rank
        # expected frequency 1/rank as per zipf’s law
        expected_frequencies.append(1 / rank)
    return actual_frequencies, expected_frequencies
 


# In[7]:


def plot(actual_frequencies, expected_frequencies):
    plt.plot(actual_frequencies, 'g*', expected_frequencies, 'ro')
    plt.show()
 
 
# We will plot the actual and expected frequencies
actual_frequencies, expected_frequencies = get_actual_and_expected_frequencies(corpus)
plot(actual_frequencies, expected_frequencies)


# #### As we can see in the above graph the two curves are almost parallel i.e we can say frequencies are proportional

# In[36]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




