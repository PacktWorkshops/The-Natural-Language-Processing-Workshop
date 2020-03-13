#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.text import Tokenizer
import numpy as np


# In[ ]:


char_tokenizer = Tokenizer(char_level=True)


# In[ ]:


text = 'The quick brown fox jumped over the lazy dog'


# In[ ]:


char_tokenizer.fit_on_texts(text)


# In[ ]:


seq = char_tokenizer.texts_to_sequences(text)


# In[ ]:


seq


# In[ ]:


char_tokenizer.sequences_to_texts(seq)


# In[ ]:


char_vectors = char_tokenizer.texts_to_matrix(text)


# In[ ]:


char_vectors


# In[ ]:


char_vectors.shape


# In[ ]:


char_vectors[0]


# In[ ]:


np.argmax(char_vectors[0])


# In[ ]:


char_tokenizer.index_word


# In[ ]:


char_tokenizer.word_index


# In[ ]:


char_tokenizer.index_word[np.argmax(char_vectors[0])]


# ### Adding this method below just for test case and is not present in the exercise

# In[ ]:


def get_one_hot_vector(text_, word_index):
    """
    >>> get_one_hot_vector("This is a cat", 0)
    [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    """
    char_tokenizer.fit_on_texts(text_)
    char_vectors = char_tokenizer.texts_to_matrix(text_)
    print(char_vectors[0])


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




