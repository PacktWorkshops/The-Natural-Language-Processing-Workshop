# coding: utf-8


import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer



corpus = ['Data Science is an overlap between Arts and Science',
          'Generally, Arts graduates are right-brained and Science graduates are left-brained',
          'Excelling in both Arts and Science at a time becomes difficult',
          'Natural Language Processing is a part of Data Science']



tfidf_model = TfidfVectorizer()
tfidf_vectors = tfidf_model.fit_transform(corpus).todense()
print(tfidf_vectors)


dump(tfidf_model, 'tfidf_model.joblib')


tfidf_model_loaded = load('tfidf_model.joblib')
loaded_tfidf_vectors = tfidf_model_loaded.transform(corpus).todense()
assert (tfidf_vectors == loaded_tfidf_vectors).all()
print(loaded_tfidf_vectors)


pickle.dump(tfidf_model, open("tfidf_model.pickle.dat", "wb"))


loaded_model = pickle.load(open("tfidf_model.pickle.dat", "rb"))
loaded_tfidf_vectors = loaded_model.transform(corpus).todense()
assert (tfidf_vectors == loaded_tfidf_vectors).all()
print(loaded_tfidf_vectors)

