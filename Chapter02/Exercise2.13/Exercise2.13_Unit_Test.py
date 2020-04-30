import pandas as pd
from IPython.core.display import display
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_text(corpus):
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


corpus = [
    'Data Science is an overlap between Arts and Science',
    'Generally, Arts graduates are right-brained and Science graduates are left-brained',
    'Excelling in both Arts and Science at a time becomes difficult',
    'Natural Language Processing is a part of Data Science']
df = vectorize_text(corpus)
display(df.head())


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


df_2 = bow_top_n(corpus, 10)
display(df_2.head())

# The below cell contains test cases of above methods

text_arr = ['However, the way we used vector values.']

# TEST 1) vectorize_text

result_df = pd.DataFrame([{'however': 1, 'the': 1, 'used': 1, 'values': 1, 'vector': 1, 'way': 1, 'we': 1}])
pd.testing.assert_frame_equal(vectorize_text(text_arr), result_df, check_names=False)

# TEST 1) bow_top_n
text_arr.extend(['This is the cat'])
result_df = pd.DataFrame(
    [{'cat': 0, 'however': 1, 'is': 0, 'the': 1, 'this': 0, 'used': 1, 'values': 1, 'vector': 1, 'way': 1, 'we': 1},
     {'cat': 1, 'however': 0, 'is': 1, 'the': 1, 'this': 1, 'used': 0, 'values': 0, 'vector': 0, 'way': 0, 'we': 0}])

pd.testing.assert_frame_equal(bow_top_n(text_arr, 10), result_df, check_names=False)
