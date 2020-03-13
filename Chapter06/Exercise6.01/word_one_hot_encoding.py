#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
data = Path('../data')
novel_lines_file = data / 'novel_lines.txt'


# In[ ]:


with novel_lines_file.open() as f:
    novel_lines_raw = f.read()


# In[ ]:


novel_lines_raw


# In[ ]:


import string
import re
 
alpha_characters = str.maketrans('', '', string.punctuation)
 
def clean_tokenize(text):
    """
    >>> clean_tokenize("This ' is a \  cat")
    ['this', 'is', 'a', 'cat']
    """
    text = text.lower()
    text = re.sub(r'\n', '*** ', text)
    text = text.translate(alpha_characters)
    text = re.sub(r' +', ' ', text)
    return text.strip().split(' ')
 
novel_lines = clean_tokenize(novel_lines_raw)


# In[ ]:


novel_lines


# In[ ]:


import numpy as np
novel_lines_array = np.array([novel_lines])
novel_lines_array = novel_lines_array.reshape(-1, 1)
novel_lines_array.shape


# In[ ]:


from sklearn import preprocessing
 
labelEncoder = preprocessing.LabelEncoder()
novel_lines_labels = labelEncoder.fit_transform(novel_lines_array)
 
import warnings
warnings.filterwarnings('ignore')
 
wordOneHotEncoder = preprocessing.OneHotEncoder()
 
line_onehot = wordOneHotEncoder.fit_transform(novel_lines_labels.reshape(-1,1))


# In[ ]:


novel_lines_labels


# In[ ]:


line_onehot


# In[ ]:


line_onehot.toarray()


# In[ ]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




