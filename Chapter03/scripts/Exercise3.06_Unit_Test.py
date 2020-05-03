# coding: utf-8

# In[6]:


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


# In[7]:


review_data = pd.read_json('../data/reviews_Musical_Instruments_5.json', lines=True)
print(review_data[['reviewText', 'overall']].head())


# In[8]:


assert review_data.shape == tuple([10261, 9])


# In[9]:


lemmatizer = WordNetLemmatizer()
review_data['cleaned_review_text'] = review_data['reviewText'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x)))]))
print(review_data[['cleaned_review_text', 'reviewText', 'overall']].head())


# In[10]:


tfidf_model = TfidfVectorizer(max_features=500)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(review_data['cleaned_review_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
assert (tfidf_df!=0).any().all()
print(tfidf_df.head())


# In[12]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(tfidf_df,review_data['overall'])
assert len(linreg.coef_) == 500
print(linreg.coef_)
print(linreg.intercept_)

print(linreg.predict(tfidf_df))


# In[21]:


review_data['predicted_score_from_linear_regression'] = linreg.predict(tfidf_df)
assert review_data['predicted_score_from_linear_regression'].min() > 0
assert review_data['predicted_score_from_linear_regression'].max() < 6
print(review_data[['overall', 'predicted_score_from_linear_regression']].head(10))
