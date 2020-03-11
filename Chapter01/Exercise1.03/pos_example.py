#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk import word_tokenize, pos_tag


# In[2]:


def get_tokens(sentence):
    '''
    This is a test case:
    >>> get_tokens('The quick brown fox jumps over the lazy dog')
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

    '''
    words = word_tokenize(sentence)
    return words


# In[3]:


words  = get_tokens("I am reading NLP Fundamentals")
print(words)


# In[4]:


def get_pos(words):
    '''
    This is a test case:
    >>> get_pos(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
    '''
    return pos_tag(words)


# In[5]:


get_pos(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])


# In[6]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:




