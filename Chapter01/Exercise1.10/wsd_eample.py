#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk.wsd import lesk
from nltk import word_tokenize


# In[3]:


sentence1 = "Keep your savings in the bank"
sentence2 = "It's so risky to drive over the banks of the road"


# In[7]:


def get_synset(sentence, word):
    """
    >>> get_synset('Dogs bark in the night', 'night')
    Synset('night.n.07')
    """
    return lesk(word_tokenize(sentence), word)
get_synset(sentence1,'bank')


# In[5]:


get_synset(sentence2,'bank')


# In[9]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




