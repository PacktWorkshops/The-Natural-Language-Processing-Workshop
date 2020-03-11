#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk import download
download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer


# In[3]:


lemmatizer = WordNetLemmatizer()


# In[18]:


def get_lemma(word):
    """
    >>> get_lemma('during')
    'during'
    """
    return lemmatizer.lemmatize(word)


# In[19]:


get_lemma('products')


# In[20]:


get_lemma('production')


# In[21]:


get_lemma('coming')


# In[22]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




