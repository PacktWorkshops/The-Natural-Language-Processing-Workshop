#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize


# In[2]:


def get_sentences(text):
    """
    >>> get_sentences("Hello, all readers. This is just test string")
    ['Hello, all readers.', 'This is just test string']
    """
    return sent_tokenize(text)
get_sentences("We are reading a book. Do you know who is the publisher? It is Packt. Packt is based out of Birmingham.")


# In[3]:


get_sentences("Mr. Donald John Trump is current president of USA. Before joining politics, he was a businessman.")


# In[4]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




