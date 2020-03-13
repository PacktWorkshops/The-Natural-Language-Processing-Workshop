#!/usr/bin/env python
# coding: utf-8

# # Text Visualization

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import re
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# In[ ]:


text = open('../data/text_corpus.txt', 'r').read()


# In[ ]:


text[:1040]


# In[ ]:


def lemmatize_and_clean(text):
    """
    >>> lemmatize_and_clean("This String is for testing and creating use cases") 
    ['this', 'string', 'is', 'for', 'testing', 'and', 'creating', 'use', 'case']
    """
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    cleaned_lemmatized_tokens = [lemmatizer.lemmatize(word.lower())                                  for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', text))]
    return cleaned_lemmatized_tokens


# In[ ]:


Counter(lemmatize_and_clean(text)).most_common(50)


# In[ ]:


stopwords = set(STOPWORDS)
cleaned_text = ' '.join(lemmatize_and_clean(text))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                max_words=50,
                stopwords = stopwords, 
                min_font_size = 10).generate(cleaned_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




