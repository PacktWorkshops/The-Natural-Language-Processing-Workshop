# coding: utf-8

# In[1]:


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


from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(tfidf_df)
reduced_tfidf = pca.transform(tfidf_df)
assert reduced_tfidf.shape[1] == 2
print(reduced_tfidf)


# In[13]:


scatter = plt.scatter(reduced_tfidf[:, 0], reduced_tfidf[:, 1], c=news_data_df['category'], cmap='gray')
plt.xlabel('dimension_1')
plt.ylabel('dimension_2')
plt.legend(handles=scatter.legend_elements()[0], labels=categories, loc='lower left')
plt.title('Representation of NEWS documents in 2D')
plt.show()


# In[ ]:




