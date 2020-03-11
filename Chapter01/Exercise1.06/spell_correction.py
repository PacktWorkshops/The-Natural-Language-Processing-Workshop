#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk import word_tokenize
from autocorrect import Speller


# In[ ]:


spell = Spellerller(lang='en')

spell('Natureal')


# In[ ]:


sentence = word_tokenize("Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes")


# In[ ]:


print(sentence)


# In[3]:


def correct_spelling(tokens):
    """
    >> correct_spelling('Ntural Luanguage')
    'Natural Language'
    """
    sentence_corrected = ' '.join([spell(word) for word in tokens])
    return sentence_corrected

    


# In[2]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:




