#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
 
from gensim.models import Doc2Vec
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords 


# In[ ]:


news_file = '../data/sample_news_data.txt'
with open(news_file, encoding="utf8", errors='ignore') as f:
    news_lines = [line for line in f.readlines()]


# In[ ]:


lines_df = pd.DataFrame()
indices  = list(range(len(news_lines)))
lines_df['news'] = news_lines
lines_df['index'] = indices


# In[ ]:


lines_df.head()


# In[ ]:


docVecModel = Doc2Vec.load('../../data/docVecModel.d2v')


# In[ ]:


from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
 
def to_vector(sentence):
    """
    >>> to_vector("US raise TV indecency US politicians are").mean()
    -0.0018694705
    """
    
    cleaned = preprocess_string(sentence)
    docVector = docVecModel.infer_vector(cleaned)
    return docVector
 
def similar_news_articles(sentence):
    """
    >>> similar_news_articles("US president in India ").index[0]
    925
    """
    vector = to_vector(sentence)
    similar_vectors = docVecModel.docvecs.most_similar(positive=[vector])
    similar_lines = lines_df[lines_df.index==similar_vectors[0][0]].news
    return similar_lines


# In[ ]:


similar_news_articles("US raise TV indecency US politicians are proposing a tough new law aimed at cracking down on indecency")


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:


to_vector("US raise TV indecency US politicians are").mean()


# In[ ]:





# In[ ]:




