#!/usr/bin/env python
# coding: utf-8

# In[3]:


from nltk import download
download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords


# In[4]:


stop_words = stopwords.words('english')


# In[5]:


print(stop_words)


# In[6]:


sentence = "I am learning Python. It is one of the most popular programming languages"
sentence_words = word_tokenize(sentence)


# In[7]:


print(sentence_words)


# In[8]:


def remove_stop_words(sentence, stop_words):
    '''
    This is a test case:
    >>> remove_stop_words(word_tokenize('The quick brown fox jumps over the lazy dog'),['The','over','the'])
    'quick brown fox jumps lazy dog'

    '''
    return ' '.join([word for word in sentence if word not in stop_words])


# In[9]:


print(remove_stop_words(sentence_words, stop_words))


# In[13]:


stop_words.extend(['I','It', 'one'])
print(remove_stop_words(sentence_words,stop_words))


# In[11]:


import doctest

doctest.testmod(verbose=True)

