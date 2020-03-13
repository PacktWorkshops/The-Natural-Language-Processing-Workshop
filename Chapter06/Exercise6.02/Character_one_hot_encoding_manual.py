#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def onehot_word(word):
    """
    >>> onehot_word('this')
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    
    """
    lookup = {v[1]: v[0] for v in enumerate(set(word))}
    #print(lookup)
    word_vector = []
    for c in word:
        one_hot_vector = [0] * len(lookup)        
        one_hot_vector[lookup[c]] = 1
        word_vector.append(one_hot_vector)
    return word_vector


# In[ ]:


onehot_vector = onehot_word('data')


# In[ ]:


print(onehot_vector)


# In[ ]:


import doctest
doctest.testmod(verbose=True)

