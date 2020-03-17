# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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


data_patio_lawn_garden = pd.read_json('data/reviews_Patio_Lawn_and_Garden_5.json', lines = True)
assert data_patio_lawn_garden.shape == tuple([13272, 9])
print(data_patio_lawn_garden[['reviewText', 'overall']].head())


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


# In[7]:


def clf_model(model_type, X_train, y):
    model = model_type.fit(X_train,y)
    predicted_labels = model.predict(tfidf_df)
    assert sorted(unique(predicted_labels)) == [0,1]
    return predicted_labels


# In[8]:


from xgboost import XGBClassifier
xgb_clf=XGBClassifier(n_estimators=20,learning_rate=0.03,max_depth=5,subsample=0.6,colsample_bytree= 0.6,reg_alpha= 10,seed=42)
data_patio_lawn_garden['predicted_labels_xgbc'] = clf_model(xgb_clf, tfidf_df, data_patio_lawn_garden['target'])
print(pd.crosstab(data_patio_lawn_garden['target'], data_patio_lawn_garden['predicted_labels_xgbc']))


# In[9]:


def reg_model(model_type, X_train, y):
    model = model_type.fit(X_train,y)
    predicted_values = model.predict(tfidf_df)
    assert min(predicted_values) > 0
    assert max(predicted_values) < 6
    return predicted_values


# In[10]:


from xgboost import XGBRegressor 
xgbr = XGBRegressor(n_estimators=20,learning_rate=0.03,max_depth=5,subsample=0.6,colsample_bytree= 0.6,reg_alpha= 10,seed=42)
data_patio_lawn_garden['predicted_values_xgbr'] = reg_model(xgbr, tfidf_df, data_patio_lawn_garden['overall'])
print(data_patio_lawn_garden[['overall', 'predicted_values_xgbr']].head(2))


# In[ ]:




