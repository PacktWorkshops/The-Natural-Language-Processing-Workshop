#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import text_to_word_sequence
from textblob import TextBlob


# In[2]:


sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performancesby Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'


# In[3]:


def get_keras_tokens(text):
    """
    >>> get_keras_tokens('This is a cat')
    ['this', 'is', 'a', 'cat']
    """
    return text_to_word_sequence(text)


# In[4]:


get_keras_tokens(sentence)


# In[5]:


def get_textblob_tokens(text):
    """
    >>> get_textblob_tokens('This is a cat')
    WordList(['This', 'is', 'a', 'cat'])
    """
    blob = TextBlob(text)
    return blob.words


# In[6]:


get_textblob_tokens(sentence)


# In[7]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:




