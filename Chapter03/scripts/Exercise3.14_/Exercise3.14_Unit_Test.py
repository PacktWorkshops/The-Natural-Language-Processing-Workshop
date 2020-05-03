# coding: utf-8


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



categories = ['misc.forsale', 'sci.electronics', 'talk.religion.misc']
news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, download_if_missing=True)


text_classifier_pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
text_classifier_pipeline.fit(news_data.data, news_data.target)
print(pd.DataFrame(text_classifier_pipeline.fit_transform(news_data.data, news_data.target).todense()).head())
