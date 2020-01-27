import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bow_text(corpus):
    """
    Will return a dataframe in which every row will ,be
    vector representation of a document in corpus
    :param corpus: input text corpus
    :return: dataframe of vectors
    """
    bag_of_words_model = CountVectorizer()

    # performs the above described three tasks on the given data corpus.
    dense_vec_matrix = bag_of_words_model.fit_transform(corpus).todense()
    bag_of_word_df = pd.DataFrame(dense_vec_matrix)
    bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
    return bag_of_word_df


def bow_top_n(corpus, n):
    """
      Will return a dataframe in which every row
      will be represented by presence or absence of top 10 most
      frequently occurring words in data corpus
      :param corpus: input text corpus
      :return: dataframe of vectors
      """
    bag_of_words_model_small = CountVectorizer(max_features=n)
    bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
    bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)
    return bag_of_word_df_small


if __name__ == '__main__':
    corpus = [
        'Data Science is an overlap between Arts and Science',
        'Generally, Arts graduates are right-brained and Science graduates are left-brained',
        'Excelling in both Arts and Science at a time becomes difficult',
        'Natural Language Processing is a part of Data Science']
    df = bow_text(corpus)
    print(df.head())
    df_2 = bow_top_n(corpus, 10)
    print(df_2.head())
