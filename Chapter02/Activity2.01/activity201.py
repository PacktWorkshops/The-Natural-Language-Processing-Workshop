#!/usr/bin/env python
# coding: utf-8

# # Extracting Top Keywords from the news article
# In this notebook, we will perform the activity of extracting top keywords from news article

# In[ ]:


import operator

from nltk.tokenize import WhitespaceTokenizer
from nltk import download, stem


# In[ ]:



# The below statement will download the stop word list
# 'nltk_data/corpora/stopwords/' at home directory of your computer
download('stopwords')
from nltk.corpus import stopwords


# In[ ]:


def load_file(file_path):
    # load the new article
    news = ''.join([line for line in open(file_path)])
    return news


# In[ ]:



def to_lower_case(text):
    """
    >>> to_lower_case("This Is a Test String")
    'this is a test string'
    """
    return text.lower()


# In[ ]:


wht = WhitespaceTokenizer()
def tokenize_text(text):
    return wht.tokenize(text=text)


# In[ ]:


stop_words = stopwords.words('english')
def remove_stop_words(token_list):
    """
    remove_stop_words(["this", "is", "a", "test", "string"])
    ['test', 'string']
    """
    return [word for word in token_list if word not in stop_words]


# In[ ]:


stemmer = stem.PorterStemmer()
def get_stems(token_list):
    """
    >>> get_stems(["this", "string", "is", "for", "testing"])
    ['thi', 'string', 'is', 'for', 'test']
    """
    return [stemmer.stem(word) for word in token_list]


# In[ ]:


def get_freq(stems):
    """
    >>> get_freq(["this", "is", "a","string","this","is","for","testing"])['is']
    2
    """
    freq_dict = {}
    for t in stems:
        freq_dict[t.strip()] = freq_dict.get(t.strip(), 0) + 1
    return freq_dict


# In[ ]:


def get_top_n_words(freq_dict, n):
    sorted_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict][:n]


# In[ ]:


path = "../data/news_article.txt"
news_article = load_file(path)


# In[ ]:


lower_case_news_art = to_lower_case(text=news_article)


# In[ ]:


tokens = tokenize_text(lower_case_news_art)


# In[ ]:


removed_tokens = remove_stop_words(tokens)


# In[ ]:


stems = get_stems(removed_tokens)


# In[ ]:


freq_dict = get_freq(stems)


# In[ ]:


top_keywords = get_top_n_words(freq_dict, 6)


# In[ ]:


top_keywords


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




