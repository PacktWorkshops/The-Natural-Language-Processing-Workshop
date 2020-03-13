#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json
import re


# In[ ]:


with open('../data/ProjectGutenbergBooks.json', 'r') as catalog_file:
        catalog = json.load(catalog_file)


# In[ ]:


catalog


# In[ ]:


GUTENBERG_URL ='https://www.gutenberg.org/files/{}/{}-0.txt'
 
def load_book(book_id):
    url = GUTENBERG_URL.format(book_id, book_id)
    contents = requests.get(url).text
    cleaned_contents = re.sub(r'\r\n', ' ', contents)
    return cleaned_contents


# In[ ]:


book_ids = [ book['id'] for book in catalog ]
books = [ load_book(id) for id in book_ids]


# In[ ]:


books[:5]


# In[ ]:


from gensim.summarization import textcleaner
from gensim.utils import simple_preprocess
 
def to_sentences(book):
    """
    >>> to_sentences("This is an example sentence. This is another example sentence.")
    [['this', 'is', 'an', 'example', 'sentence'], ['this', 'is', 'another', 'example', 'sentence']]
    """
    sentences = textcleaner.split_sentences(book)
    sentence_tokens = [simple_preprocess(sentence) for sentence in sentences]
    return sentence_tokens


# In[ ]:


books_sentences = [to_sentences(book) for book in books]
documents = [sentence for book_sent in books_sentences for sentence in book_sent]


# In[ ]:


len(documents)


# In[ ]:


from gensim.models import Word2Vec
# build vocabulary and train model
model = Word2Vec(
        documents,
        size=100,
        window=10,
        min_count=2,
        workers=10)
model.train(documents, total_examples=len(documents), epochs=50)


# In[ ]:


model.wv.most_similar(positive="worse")


# ### The below method is just for test case

# In[ ]:


def get_similar(word):
    """
    >>> get_similar('worse')
    'better'
    """
    
    return model.wv.most_similar(positive=word)[1][0]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
 
def show_vector(word):
    vector = model.wv[word]
    fig, ax = plt.subplots(1,1, figsize=(10, 2))
    ax.tick_params(axis='both', 
                   which='both',
                 left=False, 
                   bottom=False, 
                   top=False,
                   labelleft=False, 
                   labelbottom=False)
    ax.grid(False)
    print(word)
    ax.bar(range(len(vector)), vector, 0.5)
 
show_vector('sad')


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




