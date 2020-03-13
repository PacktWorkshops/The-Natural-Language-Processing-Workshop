#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import WordNetLemmatizer


# In[2]:


from nltk import word_tokenize
nltk.download('wordnet')
sentence = "The products produced by the process today are far better than what it produces generally."


# In[3]:


lemmatizer = WordNetLemmatizer()
def get_lemmas(text):
    '''
    >>> get_lemmas('why are you going there')
    'why are you going there'
    '''
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])


# In[4]:


get_lemmas(sentence)


# In[5]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




