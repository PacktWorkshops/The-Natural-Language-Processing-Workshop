#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem import RegexpStemmer


# In[2]:


# It is better to create object of RegexpStemmer here. 
#  out side method
regex_stemmer = RegexpStemmer('ing$', min=4) # creating an object of RegexpStemmer, 
                                             # any string ending with the given
                                             # regex ‘ing$’ will be removed.
def get_stems(text):
    '''
    >>> get_stems('I was going there')
    'I was go there'
    '''
    # The below code line will convert every word into its stem using regex stemmer and then join them with space.
    return ' '.join([regex_stemmer.stem(wd) for wd in text.split()])


# In[3]:


sentence = "I love playing football"
get_stems(sentence)


# In[4]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




