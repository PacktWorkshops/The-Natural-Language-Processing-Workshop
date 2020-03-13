#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
def n_gram_extractor(sentence, n):
    """
    >>> n_gram_extractor('The cute little boy',2)
    ['The', 'cute']
    ['cute', 'little']
    ['little', 'boy']
    """
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])


# In[2]:


n_gram_extractor('The cute little boy is playing with the kitten.', 2)


# In[3]:


n_gram_extractor('The cute little boy is playing with the kitten.', 3)


# In[4]:


from nltk import ngrams
list(ngrams('The cute little boy is playing with the kitten.'.split(), 2))


# In[5]:


list(ngrams('The cute little boy is playing with the kitten.'.split(), 3))


# In[6]:


from textblob import TextBlob
blob = TextBlob("The cute little boy is playing with the kitten.")
blob.ngrams(n=2)


# In[7]:


blob.ngrams(n=3)


# In[8]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:




