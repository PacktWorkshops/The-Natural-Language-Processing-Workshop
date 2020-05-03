# coding: utf-8

# In[1]:


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


data_patio_lawn_garden = pd.read_json('../data/reviews_Patio_Lawn_and_Garden_5.json', lines = True)
print(data_patio_lawn_garden[['reviewText', 'overall']].head())


# In[7]:


assert data_patio_lawn_garden.shape == tuple([13272, 9])


# In[8]:


lemmatizer = WordNetLemmatizer()


# In[9]:


data_patio_lawn_garden['cleaned_review_text'] = data_patio_lawn_garden['reviewText'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x)))]))
print(data_patio_lawn_garden[['cleaned_review_text', 'reviewText', 'overall']].head())


# In[10]:


tfidf_model = TfidfVectorizer(max_features=500)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(data_patio_lawn_garden['cleaned_review_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
assert (tfidf_df!=0).any().all()
print(tfidf_df.head())


# In[11]:


data_patio_lawn_garden['target'] = data_patio_lawn_garden['overall'].apply(lambda x : 0 if x<=4 else 1)
assert sorted(data_patio_lawn_garden['target'].unique()) == [0,1]
print(data_patio_lawn_garden['target'].value_counts())


# In[12]:


from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(tfidf_df, data_patio_lawn_garden['target'])
data_patio_lawn_garden['predicted_labels_dtc'] = dtc.predict(tfidf_df)
assert sorted(data_patio_lawn_garden['predicted_labels_dtc'].unique()) == [0,1]


# In[13]:


print(pd.crosstab(data_patio_lawn_garden['target'], data_patio_lawn_garden['predicted_labels_dtc']))


# In[17]:


from sklearn import tree
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(tfidf_df, data_patio_lawn_garden['overall'])
data_patio_lawn_garden['predicted_values_dtr'] = dtr.predict(tfidf_df)

assert data_patio_lawn_garden['predicted_values_dtr'].min() > 0
assert data_patio_lawn_garden['predicted_values_dtr'].max() < 6
print(data_patio_lawn_garden[['predicted_values_dtr', 'overall']].head(10))
