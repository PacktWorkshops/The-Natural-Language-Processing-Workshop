#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


def get_tf_idf_vectors(corpus):
    """
    >>> float(get_tf_idf_vectors(["This is a sample text for testing", "This is again a sample text for testing"]).mean(axis=1).mean(axis=0)[0][0])
    0.3622687874257512
    """
    tfidf_model = TfidfVectorizer()
    vector_list = tfidf_model.fit_transform(corpus).todense()
    return vector_list


# In[ ]:


corpus = [
        'Data Science is an overlap between Arts and Science',
        'Generally, Arts graduates are right-brained and Science graduates are left-brained',
        'Excelling in both Arts and Science at a time becomes difficult',
        'Natural Language Processing is a part of Data Science'
    ]


# In[ ]:


vector_list = get_tf_idf_vectors(corpus)
print(vector_list)


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




