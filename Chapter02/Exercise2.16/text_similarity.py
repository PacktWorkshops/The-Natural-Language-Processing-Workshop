import unittest

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()
import numpy as np


def extract_text_similarity_jaccard(text1, text2):
   
    lemmatizer = WordNetLemmatizer()

    words_text1 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1)]
    words_text2 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2)]
    nr = len(set(words_text1).intersection(set(words_text2)))
    dr = len(set(words_text1).union(set(words_text2)))
    jaccard_sim = nr / dr
    return jaccard_sim


pair1 = ["What you do defines you", "Your deeds define you"]
pair2 = ["Once upon a time there lived a king.", "Who is your queen?"]
pair3 = ["He is desperate", "Is he not desperate?"]

extract_text_similarity_jaccard(pair1[0], pair1[1])

extract_text_similarity_jaccard(pair2[0], pair2[1])

extract_text_similarity_jaccard(pair3[0], pair3[1])


def get_tf_idf_vectors(corpus):
  
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_results = tfidf_vectorizer.fit_transform(corpus).todense()
    return tfidf_results


corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]
tf_idf_vectors = get_tf_idf_vectors(corpus)

print(cosine_similarity(tf_idf_vectors[0], tf_idf_vectors[1]))

print(cosine_similarity(tf_idf_vectors[2], tf_idf_vectors[3]))

print(cosine_similarity(tf_idf_vectors[4], tf_idf_vectors[5]))


class TestMethods(unittest.TestCase):

    def test_cosine_similarity(self):
        result = 0.5773502691896258

        self.assertEqual(
            cosine_similarity(np.array([1, 1, 1]).reshape(1, -1), np.array([0, 0, 1]).reshape(1, -1))[0][0],
            result)


def test_get_tf_idf_vectors(self):
    result = 0.36795725772534665
    self.assertEqual(float(
        get_tf_idf_vectors(["This is a test String", "This is an another test String"]).mean(axis=0).mean(axis=1)),
        result)


if __name__ == '__main__':
    unittest.main()
