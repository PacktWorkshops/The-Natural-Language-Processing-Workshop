# # Implementing lesk algorithm from scratch using string similarity and text vectorization
import unittest

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tf_idf_vectors(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_results = tfidf_vectorizer.fit_transform(corpus).todense()
    return tfidf_results


def to_lower_case(corpus):
    lowercase_corpus = [x.lower() for x in corpus]
    return lowercase_corpus


def find_sentence_defnition(sent_vector, defnition_vectors):
    """
    
    This method will find cosine similarity of sentence with
    the possible definitions and return the one with highest similarity score
    along with the similarity score.
    
    """
    result_dict = {}
    for defnition_id, def_vector in defnition_vectors.items():
        sim = cosine_similarity(sent_vector, def_vector)
        result_dict[defnition_id] = sim[0][0]
    defnition = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[0]
    return defnition[0], defnition[1]


corpus = ["On the banks of river Ganga, there lies the scent of spirituality",
          "An institute where people can store extra cash or money.",
          "The land alongside or sloping down to a river or lake"
          "What you do defines you",
          "Your deeds define you",
          "Once upon a time there lived a king.",
          "Who is your queen?",
          "He is desperate",
          "Is he not desperate?"]

lower_case_corpus = to_lower_case(corpus)
corpus_tf_idf = get_tf_idf_vectors(lower_case_corpus)
sent_vector = corpus_tf_idf[0]
defnition_vectors = {'def1': corpus_tf_idf[1], 'def2': corpus_tf_idf[2]}
defnition_id, score = find_sentence_defnition(sent_vector, defnition_vectors)
print("The defnition of word {} is {} with similarity of {}".format('bank', defnition_id, score))


class TestMethods(unittest.TestCase):

    def test_find_sentence_defnition(self):
        result = ('def2', 0.8660254037844388)
        self.assertEqual(find_sentence_defnition(np.array([1, 1, 1, 0]).reshape(1, -1),
                                                 {'def1': np.array([1, 1, 0, 0]).reshape(1, -1),
                                                  'def2': np.array([1, 1, 1, 1]).reshape(1, -1)}), result)

    def test_get_tf_idf_vectors(self):
        result = 0.36795725772534665
        self.assertEqual(float(
            get_tf_idf_vectors(["This is a test String", "This is an another test String"]).mean(axis=0).mean(axis=1)),
                         result)


if __name__ == '__main__':
    unittest.main()
