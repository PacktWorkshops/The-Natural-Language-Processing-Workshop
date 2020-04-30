import unittest

from sklearn.feature_extraction.text import TfidfVectorizer


def get_tf_idf_vectors(corpus):

    tfidf_model = TfidfVectorizer()
    vector_list = tfidf_model.fit_transform(corpus).todense()
    return vector_list


corpus = [
    'Data Science is an overlap between Arts and Science',
    'Generally, Arts graduates are right-brained and Science graduates are left-brained',
    'Excelling in both Arts and Science at a time becomes difficult',
    'Natural Language Processing is a part of Data Science'
]

vector_list = get_tf_idf_vectors(corpus)
print(vector_list)


class TestMethods(unittest.TestCase):

    def test_get_tf_idf_vectors(self):
        result = 0.3622687874257512
        self.assertEqual(float(float(get_tf_idf_vectors(["This is a sample text for testing", "This is again a sample text for testing"]).mean(axis=1).mean(axis=0)[0][0])),
            result)


if __name__ == '__main__':
    unittest.main()
