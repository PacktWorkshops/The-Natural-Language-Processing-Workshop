import nltk

# The below statement will download the stop word list
# 'nltk_data/corpora/stopwords/' at home directory of your computer
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
"""
This code will remove stop words from english sentence.
"""
# The below line will load english stopword list from corpora
stop_words = stopwords.words('english')
print(stop_words)

sentence = "I am learning Python. It is one of the most popular programming languages"
sentence_words = word_tokenize(sentence)
print(sentence_words)

sentence_no_stops = ' '.join([word for word in sentence_words if word not in stop_words])
print(sentence_no_stops)

