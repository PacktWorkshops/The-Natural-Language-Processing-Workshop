#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk import download
from nltk import pos_tag
from nltk import ne_chunk
from nltk import word_tokenize
download('maxent_ne_chunker')
download('words')


# In[2]:


sentence = "We are reading a book published by Packt which is based out of Birmingham."


# In[9]:


def get_ner(text):
    """
    >>> get_ner("India is the second most populous country")
    [Tree('NE', [('India', 'NNP')])]
    """
    i = ne_chunk(pos_tag(word_tokenize(text)), binary=True)
    return [a for a in i if len(a)==1]
get_ner(sentence)


# In[10]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




