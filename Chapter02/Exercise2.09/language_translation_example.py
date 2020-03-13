#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob


# In[8]:


def translate(text,from_l,to_l):
    '''
    >>> translate('Hello','en', 'es')
    TextBlob("Hola")
    '''
    en_blob = TextBlob(text)
    return en_blob.translate(from_lang=from_l, to=to_l)


# In[9]:


translate(text='muy bien',from_l='es',to_l='en')


# In[10]:


translate('Hello','en', 'es')


# In[11]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




