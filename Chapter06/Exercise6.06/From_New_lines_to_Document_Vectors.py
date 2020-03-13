#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
import random
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


sample_news_data = '../data/sample_news_data.txt'


# In[ ]:


with open(sample_news_data, encoding="utf8", errors='ignore') as f:
    news_lines = [line for line in f.readlines()]


# In[ ]:


lines_df = pd.DataFrame()


# In[ ]:


indices  = list(range(len(news_lines)))


# In[ ]:


lines_df['news'] = news_lines
lines_df['index'] = indices


# In[ ]:


lines_df.head()


# In[ ]:


def preprocess( document):
    """
    >>> preprocess("There is no list, successful applicants will be called for personal interview")
    ['list', 'success', 'applic', 'call', 'person', 'interview']
    """
    return preprocess_string(remove_stopwords(document))


# In[ ]:


document = lines_df['news'].apply(preprocess)


# In[ ]:


documents = [ TaggedDocument( text, [index]) 
                          for index, text in document.iteritems() ]


# In[ ]:


class DocumentDataset(object):
    
    def __init__(self, data:pd.DataFrame, column):
        document = data[column].apply(self.preprocess)
        
        self.documents = [ TaggedDocument( text, [index]) 
                          for index, text in document.iteritems() ]
      
    def preprocess(self, document):
        return preprocess_string(remove_stopwords(document))
        
    def __iter__(self):
        for document in self.documents:
            yield documents
            
    def tagged_documents(self, shuffle=False):
        if shuffle:
            random.shuffle(self.documents)
        return self.documents


# In[ ]:


documents_dataset = DocumentDataset(lines_df, 'news')


# In[ ]:


docVecModel = Doc2Vec(min_count=1, window=5, vector_size=100, sample=1e-4, negative=5, workers=8)
docVecModel.build_vocab(documents_dataset.tagged_documents())


# In[ ]:


docVecModel.train(documents_dataset.tagged_documents(shuffle=True),
            total_examples = docVecModel.corpus_count,
           epochs=10)


# In[ ]:


docVecModel.save('../data/docVecModel.d2v')


# In[ ]:


docVecModel[657]


# In[ ]:


def get_document_vector(document_index):
    """
    >>> get_document_vector(455).max()
    0.85495985
    
    """
    return docVecModel[document_index]


# In[ ]:


import matplotlib.pyplot as plt
 
def show_image(vector, line):
    fig, ax = plt.subplots(1,1, figsize=(10, 2))
    ax.tick_params(axis='both', 
                   which='both',
                   left=False, 
                   bottom=False,
                   top=False,
                   labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    print(line)
    ax.bar(range(len(vector)), vector, 0.5)
   
  
def show_news_line(line_number):
    line = lines_df[lines_df.index==line_number].news
    doc_vector = docVecModel[line_number]
    show_image(doc_vector, line)


# In[ ]:


show_news_line(872)


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




