#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk.stem.porter import *


# In[3]:


sentence = "Before eating, it would be nice to sanitize your hands with a sanitizer"


# In[13]:


# It is better to create object of PorterStemmer here. 
#  out side method
ps_stemmer = PorterStemmer()
def get_stems(text):
    '''
    >>> get_stems('Why are you doing this.')
    'whi are you do this.'
    '''
    return ' '.join([ps_stemmer.stem(wd) for wd in text.split()])


# In[14]:


get_stems(sentence)


# In[15]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




