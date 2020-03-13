#!/usr/bin/env python
# coding: utf-8

# # Implementing lesk algorithm from scratch using string similarity and text vectorization

# In[ ]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np


# In[ ]:


def get_tf_idf_vectors(corpus):
    """
    >>> float(get_tf_idf_vectors(["This is a test String", "This is an another test String"]).mean(axis=0).mean(axis=1))
    0.36795725772534665
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_results = tfidf_vectorizer.fit_transform(corpus).todense()
    return tfidf_results


# In[ ]:


def to_lower_case(corpus):
    lowercase_corpus = [x.lower() for x in corpus]
    return lowercase_corpus


# In[ ]:


def find_sentence_defnition(sent_vector,defnition_vectors):
    """
    
    This method will find cosine similarity of sentence with
    the possible definitionsdefnitions and return the one with highest similarity score
    along with the similarity score.
    
    >>> find_sentence_defnition(np.array([1,1,1, 0]).reshape(1,-1),\
                       {'def1':np.array([1,1, 0, 0]).reshape(1,-1), 'def2':np.array([1,1, 1, 1]).reshape(1,-1)})
    ('def2', 0.8660254037844388)
    
    """
    result_dict = {}
    for defnition_id,def_vector in defnition_vectors.items():
        sim = cosine_similarity(sent_vector,def_vector)
        result_dict[defnition_id] = sim[0][0]
    defnition  = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[0]
    return defnition[0],defnition[1]


# In[ ]:


corpus = ["On the banks of river Ganga, there lies the scent of spirituality",
          "An institute where people can store extra cash or money.",
          "The land alongside or sloping down to a river or lake"
           "What you do defines you",
           "Your deeds define you",
           "Once upon a time there lived a king.",
           "Who is your queen?",
            "He is desperate",
           "Is he not desperate?"]


# In[ ]:


lower_case_corpus  = to_lower_case(corpus)
corpus_tf_idf  = get_tf_idf_vectors(lower_case_corpus)
sent_vector = corpus_tf_idf[0]
defnition_vectors = {'def1':corpus_tf_idf[1],'def2':corpus_tf_idf[2]}
defnition_id, score  = find_sentence_defnition(sent_vector,defnition_vectors)
print("The defnition of word {} is {} with similarity of {}".format('bank',defnition_id,score))


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




