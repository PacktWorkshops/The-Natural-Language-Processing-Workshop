#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[38]:


def clean_text(sentence):
    '''
    This is a test case:
    >>> clean_text('The quick brown, fox jumps over ., the lazy dog')
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

    '''
    return re.sub(r'([^\s\w]|_)+', ' ', sentence).split()


# In[39]:


sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'
clean_text(sentence)


# In[40]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




