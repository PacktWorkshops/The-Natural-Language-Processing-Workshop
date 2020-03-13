#!/usr/bin/env python
# coding: utf-8

# In[5]:


from nltk import word_tokenize
sentence = "She sells seashells on the seashore"


# In[6]:


def remove_stop_words(text,stop_word_list):
    '''
    >>> remove_stop_words('I am doing it',['it'])
    'I am doing'
    '''
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_word_list])


# In[7]:


custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']
remove_stop_words(sentence,custom_stop_word_list)


# In[8]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




