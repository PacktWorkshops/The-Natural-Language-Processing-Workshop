#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import zipfile


# In[ ]:


GLOVE_DIR = '../data/glove/'
GLOVE_ZIP = GLOVE_DIR + 'glove6b50dtxt.zip'
print(GLOVE_ZIP)
 

zip_ref = zipfile.ZipFile(GLOVE_ZIP, 'r')
zip_ref.extractall(GLOVE_DIR)
zip_ref.close()


# In[ ]:


def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open( fn,'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model
 
glove_vectors = load_glove_vectors(GLOVE_DIR +'glove.6B.50d.txt')


# In[ ]:


glove_vectors


# In[ ]:


glove_vectors["dog"]


# In[ ]:


glove_vectors["cat"]


# ### the below method is just for test case and is not in exercise

# In[ ]:


def get_vector(word):
    """
    >>> get_vector("focus").mean()
    0.003364140000000012
    """
    return glove_vectors[word]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
 
def to_vector(glove_vectors, word):
    vector = glove_vectors.get(word.lower())
    if vector is None:
        vector = [0] * 50
    return vector 
 
def to_image(vector, word=''):
    fig, ax = plt.subplots(1,1)
    ax.tick_params(axis='both', which='both',
                   left=False, 
                   bottom=False, 
                   top=False,
                   labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.bar(range(len(vector)), vector, 0.5)
    ax.text(s=word, x=1, y=vector.max()+0.5)
    return vector


# In[ ]:


man = to_image(to_vector(glove_vectors, "man"))


# In[ ]:


woman = to_image(to_vector(glove_vectors, "woman"))


# In[ ]:


king = to_image(to_vector(glove_vectors, "king"))


# In[ ]:


queen = to_image(to_vector(glove_vectors, "queen"))


# In[ ]:


diff = to_image(king - man + woman - queen)


# In[ ]:


nd = to_image(king - man + woman)


# In[ ]:


import doctest
doctest.testmod(verbose=True)


# In[ ]:




