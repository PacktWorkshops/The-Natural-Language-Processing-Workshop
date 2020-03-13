#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
sentence = TextBlob('She sells seashells on the seashore')


# In[2]:


sentence.words


# In[3]:


def singularize(word):
    '''
    >>> singularize(sentence.words[2])
    'seashell'
    '''
    return word.singularize()


# In[4]:


singularize(sentence.words[2])


# In[5]:


def pillularize(word):
    '''
    >>> pillularize(sentence.words[5])
    'seashores'
    '''
    return word.pluralize()


# In[6]:


pillularize(sentence.words[5])


# In[7]:


import doctest
doctest.testmod(verbose=True)

