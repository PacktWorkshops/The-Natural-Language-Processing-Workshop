# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from pylab import *
import nltk
import warnings
warnings.filterwarnings('ignore')


# In[2]:


review_data = pd.read_json('../data/reviews_Musical_Instruments_5.json', lines=True)
print(review_data[['reviewText', 'overall']].head())


# In[3]:


assert review_data.shape == tuple([10261, 9])


# In[4]:


lemmatizer = WordNetLemmatizer()
review_data['cleaned_review_text'] = review_data['reviewText'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x)))]))


# In[5]:


print(review_data[['cleaned_review_text', 'reviewText', 'overall']].head())


# In[6]:


tfidf_model = TfidfVectorizer(max_features=500)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(review_data['cleaned_review_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
print(tfidf_df.head())


# In[7]:


assert (tfidf_df!=0).any().all()


# In[8]:


review_data['target'] = review_data['overall'].apply(lambda x : 0 if x<=4 else 1)
print(review_data['target'].value_counts())


# In[9]:


assert sorted(review_data['target'].unique()) == [0,1]


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(tfidf_df,review_data['target'])
review_data['predicted_labels_knn'] = knn.predict(tfidf_df)
assert sorted(review_data['predicted_labels_knn'].unique()) == [0,1]
print(pd.crosstab(review_data['target'], review_data['predicted_labels_knn']))
