#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk import stem


# In[2]:


def get_stems(word,stemmer):
    """
    Test doc
    >>> porterStem = stem.PorterStemmer()
    >>> get_stems('during',porterStem)
    'dure'
    """
    return stemmer.stem(word)


# In[3]:


porterStem = stem.PorterStemmer()


# In[4]:


get_stems("production",porterStem)


# In[5]:


get_stems("coming",porterStem)


# In[6]:


get_stems("firing",porterStem)


# In[7]:


get_stems("battling",porterStem)


# In[8]:


snowball_stemmer = stem.SnowballStemmer("english")


# In[9]:


get_stems("battling",snowball_stemmer)


# In[10]:


import doctest

doctest.testmod(verbose=True)

