#!/usr/bin/env python
# coding: utf-8

# In[6]:


from nltk import word_tokenize


# In[7]:


def get_tokens(sentence):
    '''
    This is a test case:
    >>> get_tokens('The quick brown fox jumps over the lazy dog')
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

    '''
    words = word_tokenize(sentence)
    return words


# In[8]:


print(get_tokens("I am reading NLP Fundamentals."))


# In[9]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:




