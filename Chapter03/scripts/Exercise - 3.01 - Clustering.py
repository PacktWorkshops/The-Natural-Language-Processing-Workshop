# coding: utf-8


from sklearn.datasets import fetch_20newsgroups
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib as mpl
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity
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


nltk.download('stopwords')
stop_words=stopwords.words('english')
stop_words=stop_words+list(string.printable)
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()


# In[3]:


categories= ['misc.forsale', 'sci.electronics', 'talk.religion.misc']


# In[4]:


news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, download_if_missing=True)


# In[5]:


print(news_data['data'][:5])


# In[45]:


print(news_data.target)
assert sorted(list(unique(news_data.target))) == sorted([0, 1, 2])


# In[7]:


news_data_df = pd.DataFrame({'text' : news_data['data'], 'category': news_data.target})
print(news_data_df.head())


# In[13]:


assert news_data_df.shape == tuple([1553, 2])


# In[14]:


print(news_data_df['category'].value_counts())


# In[15]:


nltk.download('punkt')


# In[16]:


news_data_df['cleaned_text'] = news_data_df['text'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if word.lower() not in stop_words]))


# In[17]:


tfidf_model = TfidfVectorizer(max_features=200)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(news_data_df['cleaned_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
print(tfidf_df.head())


# In[23]:


assert (tfidf_df!=0).any().all()


# In[26]:


from sklearn.metrics.pairwise import euclidean_distances as euclidean
dist = 1 - euclidean(tfidf_df)


# In[33]:


assert (dist!=0).any()


# In[38]:


import scipy.cluster.hierarchy as sch


# In[28]:


dendrogram = sch.dendrogram(sch.linkage(dist, method='ward'))

plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.title('Dendrogram')
plt.show()


# In[39]:


k=4
clusters = fcluster(sch.linkage(dist, method='ward'), k, criterion='maxclust')
print(clusters)


# In[48]:


assert sorted(list(unique(clusters))) == sorted([1, 2, 3, 4])


# In[31]:


news_data_df['obtained_clusters'] = clusters
print(pd.crosstab(news_data_df['category'].replace({0:'misc.forsale', 1:'sci.electronics', 2:'talk.religion.misc'}),            news_data_df['obtained_clusters'].            replace({1 : 'cluster_1', 2 : 'cluster_2', 3 : 'cluster_3', 4: 'cluster_4'})))

