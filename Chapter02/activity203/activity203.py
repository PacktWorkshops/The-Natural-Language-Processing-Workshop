import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import re
import string
import pandas as pd


newsgroups_data_sample = fetch_20newsgroups(subset='train')
lemmatizer = WordNetLemmatizer()


newsgroups_text_df = pd.DataFrame({'text' : newsgroups_data_sample['data']})
newsgroups_text_df.head()


stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)


newsgroups_text_df['cleaned_text'] = newsgroups_text_df['text'].apply(\
lambda x : ' '.join([lemmatizer.lemmatize(word.lower()) \
    for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if word.lower() not in stop_words]))

bag_of_words_model = CountVectorizer(max_features= 20)
bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(newsgroups_text_df['cleaned_text']).todense())
bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()

tfidf_model = TfidfVectorizer(max_features=20)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(newsgroups_text_df['cleaned_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
tfidf_df.head()


