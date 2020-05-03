# coding: utf-8

from sklearn.datasets import fetch_20newsgroups
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from pylab import *
import nltk
import warnings
warnings.filterwarnings('ignore')


# In[2]:


stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)
lemmatizer = WordNetLemmatizer()


# In[3]:


categories= ['misc.forsale', 'sci.electronics', 'talk.religion.misc']


# In[4]:


news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, download_if_missing=True)
news_data_df = pd.DataFrame({'text' : news_data['data'], 'category': news_data.target})
assert sorted(list(unique(news_data.target))) == sorted([0, 1, 2])
print(news_data_df.head())


# In[5]:


news_data_df['cleaned_text'] = news_data_df['text'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if word.lower() not in stop_words]))


# In[6]:


tfidf_model = TfidfVectorizer(max_features=20)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(news_data_df['cleaned_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
assert (tfidf_df!=0).any().all()
print(tfidf_df.head())


# In[7]:


correlation_matrix = tfidf_df.corr()
assert (correlation_matrix!=0).any().all()
print(correlation_matrix.head())


# In[11]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlation_matrix,annot=True, fmt='.1g', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# In[22]:


import numpy as np
correlation_matrix_ut = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape)).astype(np.bool))
correlation_matrix_melted = correlation_matrix_ut.stack().reset_index()
assert correlation_matrix_melted.shape[1] == 3
correlation_matrix_melted.columns = ['word1', 'word2', 'correlation']
print(correlation_matrix_melted[(correlation_matrix_melted['word1']!=                           correlation_matrix_melted['word2']) & (correlation_matrix_melted['correlation']>.7)])


# In[28]:


tfidf_df_without_correlated_word = tfidf_df.drop(['nntp', 'posting', 'organization'], axis = 1)
assert tfidf_df_without_correlated_word.shape == tuple([1553, 17])
print(tfidf_df_without_correlated_word.head())


# In[ ]:




