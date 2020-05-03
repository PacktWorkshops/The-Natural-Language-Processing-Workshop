# coding: utf-8

# In[2]:


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


# In[3]:


data_patio_lawn_garden = pd.read_json('../data/reviews_Patio_Lawn_and_Garden_5.json', lines = True)
assert data_patio_lawn_garden.shape == tuple([13272, 9])
data_patio_lawn_garden[['reviewText', 'overall']].head()


# In[3]:


lemmatizer = WordNetLemmatizer()


# In[4]:


data_patio_lawn_garden['cleaned_review_text'] = data_patio_lawn_garden['reviewText'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x)))]))
print(data_patio_lawn_garden[['cleaned_review_text', 'reviewText', 'overall']].head())


# In[5]:


tfidf_model = TfidfVectorizer(max_features=500)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(data_patio_lawn_garden['cleaned_review_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
assert (tfidf_df!=0).any().all()
print(tfidf_df.head())


# In[6]:


data_patio_lawn_garden['target'] = data_patio_lawn_garden['overall'].apply(lambda x : 0 if x<=4 else 1)
assert sorted(data_patio_lawn_garden['target'].unique()) == [0,1]
print(data_patio_lawn_garden['target'].value_counts())


# In[9]:


def clf_model(model_type, X_train, y):
    model = model_type.fit(X_train,y)
    predicted_labels = model.predict(tfidf_df)
    assert sorted(unique(predicted_labels)) == [0,1]
    return predicted_labels


# In[10]:


from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators=20,max_depth=4,max_features='sqrt',random_state=1)
data_patio_lawn_garden['predicted_labels_rfc'] = clf_model(rfc, tfidf_df, data_patio_lawn_garden['target'])
print(pd.crosstab(data_patio_lawn_garden['target'], data_patio_lawn_garden['predicted_labels_rfc']))


# In[11]:


def reg_model(model_type, X_train, y):
    model = model_type.fit(X_train,y)
    predicted_values = model.predict(tfidf_df)
    assert min(predicted_values) > 0
    assert max(predicted_values) < 6
    return predicted_values


# In[12]:


from sklearn.ensemble import RandomForestRegressor 
rfg = RandomForestRegressor(n_estimators=20,max_depth=4,max_features='sqrt',random_state=1)
data_patio_lawn_garden['predicted_values_rfg'] = reg_model(rfg, tfidf_df, data_patio_lawn_garden['overall'])
print(data_patio_lawn_garden[['overall', 'predicted_values_rfg']].head(10))


# In[ ]:




