#!/usr/bin/env python
# coding: utf-8

# In[2]:


sentence = "I visited US from UK on 22-10-18"


# In[14]:


def normalize(text):
    '''
    This is a test case:
    >>> normalize('US and UK are two superpowers')
    'United States and United Kingdom are two superpowers'

    '''
    return text.replace("US", "United States").replace("UK", "United Kingdom").replace("-18", "-2018")


# In[15]:


normalized_sentence = normalize(sentence)
print(normalized_sentence)


# In[16]:


normalized_sentence = normalize('US and UK are two superpowers')
print(normalized_sentence)


# In[17]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




