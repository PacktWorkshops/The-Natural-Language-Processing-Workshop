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



review_data = pd.read_json('../data/reviews_Musical_Instruments_5.json', lines=True)
print(review_data[['reviewText', 'overall']].head())


assert review_data.shape == tuple([10261, 9])

lemmatizer = WordNetLemmatizer()
review_data['cleaned_review_text'] = review_data['reviewText'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word.lower())     for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x)))]))


print(review_data[['cleaned_review_text', 'reviewText', 'overall']].head())


tfidf_model = TfidfVectorizer(max_features=500)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(review_data['cleaned_review_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
print(tfidf_df.head())


assert (tfidf_df!=0).any().all()


review_data['target'] = review_data['overall'].apply(lambda x : 0 if x<=4 else 1)
print(review_data['target'].value_counts())


assert sorted(review_data['target'].unique()) == [0,1]


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(tfidf_df,review_data['target'])
predicted_labels = logreg.predict(tfidf_df)
print(logreg.predict_proba(tfidf_df)[:,1])


review_data['predicted_labels'] = predicted_labels
assert sorted(review_data['predicted_labels'].unique()) == [0,1]
print(pd.crosstab(review_data['target'], review_data['predicted_labels']))
