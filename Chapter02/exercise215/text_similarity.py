from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_similarity_jaccard(text1, text2):
    """
    This method will return Jaccard similarity between two texts
    after lemmatizing them.
    :param text1: text1
    :param text2: text2
    :return: similarity measure
    """
    lemmatizer = WordNetLemmatizer()

    words_text1 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1)]
    words_text2 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2)]
    nr = len(set(words_text1).intersection(set(words_text2)))
    dr = len(set(words_text1).union(set(words_text2)))
    jaccard_sim = nr / dr
    return jaccard_sim


def get_tf_idf_vectors(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_results = tfidf_vectorizer.fit_transform(corpus).todense()
    return tfidf_results


if __name__ == '__main__':
    pair1 = ["What you do defines you", "Your deeds define you"]
    pair2 = ["Once upon a time there lived a king.", "Who is your queen?"]
    pair3 = ["He is desperate", "Is he not desperate?"]

    print(extract_text_similarity_jaccard(pair1[0], pair1[1]))
    print(extract_text_similarity_jaccard(pair2[0], pair2[1]))
    print(extract_text_similarity_jaccard(pair3[0], pair3[1]))

    corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]
    tf_idf_vectors = get_tf_idf_vectors(corpus)
    print(cosine_similarity(tf_idf_vectors[0], tf_idf_vectors[1]))
    print(cosine_similarity(tf_idf_vectors[2], tf_idf_vectors[3]))
    print(cosine_similarity(tf_idf_vectors[4], tf_idf_vectors[5]))


