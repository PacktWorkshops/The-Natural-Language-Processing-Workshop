import pandas as pd
from string import punctuation
import nltk
from IPython.core.display import display

nltk.download('tagsets')
from nltk.data import load

nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk import word_tokenize
from collections import Counter


def get_tagsets():
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    return list(tagdict.keys())


tag_list = get_tagsets()

print(tag_list)


# This method will count occurrence of pos tags in each sentence.
def get_pos_occurrence_freq(data, tag_list):
    # Get list of sentences in text_list
    text_list = data.text

    # create empty dataframe
    feature_df = pd.DataFrame(columns=tag_list)
    for text_line in text_list:
        # get pos tags of each word.
        pos_tags = [j for i, j in pos_tag(word_tokenize(text_line))]

        # create a dict of pos tags and their frequency in given sentence.
        row = dict(Counter(pos_tags))
        feature_df = feature_df.append(row, ignore_index=True)
    feature_df.fillna(0, inplace=True)
    return feature_df


data = pd.read_csv('../data/data.csv', header=0)
feature_df = get_pos_occurrence_freq(data, tag_list)
display(feature_df.head())


def add_punctuation_count(feature_df, data):
    # The below code line will find the intersection of set
    # of punctuations in text and punctuation set
    # imported from string module of python and find the length of
    # intersection set in each row and add it to column `num_of_unique_punctuations`
    # of data frame.

    feature_df['num_of_unique_punctuations'] = data['text'].apply(lambda x: len(set(x).intersection(set(punctuation))))
    return feature_df


feature_df = add_punctuation_count(feature_df, data)

display(feature_df['num_of_unique_punctuations'].head())


def get_capitalized_word_count(feature_df, data):
    # The below code line will tokenize text in every row and
    # create a set of only capital words, then find the length of
    # this set and add it to the column `number_of_capital_words`
    # of dataframe.

    feature_df['number_of_capital_words'] = data['text'].apply(
        lambda x: len([word for word in word_tokenize(str(x)) if word[0].isupper()]))
    return feature_df


feature_df = get_capitalized_word_count(feature_df, data)

display(feature_df['number_of_capital_words'].head())


def get_small_word_count(feature_df, data):
    # The below code line will tokenize text in every row and
    # create a set of only small words, then find the length of
    # this set and add it to the column `number_of_small_words`
    # of dataframe.

    feature_df['number_of_small_words'] = data['text'].apply(
        lambda x: len([word for word in word_tokenize(str(x)) if word[0].islower()]))
    return feature_df


feature_df = get_small_word_count(feature_df, data)
display(feature_df['number_of_small_words'].head())


def get_number_of_alphabets(feature_df, data):
    # The below code line will break the text line in a list of
    # characters in each row and add the count of that list into
    # the columns `number_of_alphabets`

    feature_df['number_of_alphabets'] = data['text'].apply(lambda x: len([ch for ch in str(x) if ch.isalpha()]))
    return feature_df


feature_df = get_number_of_alphabets(feature_df, data)
display(feature_df['number_of_alphabets'].head())


def get_number_of_digit_count(feature_df, data):
    # The below code line will break the text line in a list of
    # digits in each row and add the count of that list into
    # the columns `number_of_digits`

    feature_df['number_of_digits'] = data['text'].apply(lambda x: len([ch for ch in str(x) if ch.isdigit()]))
    return feature_df


feature_df = get_number_of_digit_count(feature_df, data)
display(feature_df['number_of_digits'].head())


def get_number_of_words(feature_df, data):
    # The below code line will break the text line in a list of
    # words in each row and add the count of that list into
    # the columns `number_of_digits`

    feature_df['number_of_words'] = data['text'].apply(lambda x
                                                       : len(word_tokenize(str(x))))

    return feature_df


feature_df = get_number_of_words(feature_df, data)
display(feature_df['number_of_words'].head())


def get_number_of_whitespaces(feature_df, data):
    # The below code line will generate list of white spaces
    # in each row and add the length of that list into
    # the columns `number_of_white_spaces`

    feature_df['number_of_white_spaces'] = data['text'].apply(lambda x: len([ch for ch in str(x) if ch.isspace()]))

    return feature_df


feature_df = get_number_of_whitespaces(feature_df, data)
display(feature_df['number_of_white_spaces'].head())

display(feature_df.head())

# Create a test data frame
test_df = pd.DataFrame([{"text": "this is  a cat"}, {"text": "why are you so happy"}])
tag_list = ['JJR', 'CC', 'VBN', 'CD', 'NNS']

# 1) TEST get_pos_occurrence_freq()
test_feature_df = get_pos_occurrence_freq(test_df, tag_list)
result_df = pd.DataFrame([{"JJR": 0, "CC": 0, "VBN": 0, "CD": 0, "NNS": 0,
                           "DT": 2.0, "NN": 1.0, "VBZ": 1.0, "JJ": 0.0, "PRP": 0.0, "RB": 0.0, "VBP": 0.0, "WRB": 0.0},
                          {"JJR": 0, "CC": 0, "VBN": 0, "CD": 0, "NNS": 0,
                           "DT": 0.0, "NN": 0.0, "VBZ": 0.0, "JJ": 1.0, "PRP": 1.0, "RB": 1.0, "VBP": 1.0, "WRB": 1.0}])

# Assert equality of the results and expected data frames

pd.testing.assert_frame_equal(test_feature_df, result_df, check_names=False, check_like=True)

# 2) TEST add_punctuation_count()
add_punctuation_count(test_feature_df, test_df)

result_df['num_of_unique_punctuations'] = [0, 0]
pd.testing.assert_frame_equal(add_punctuation_count(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 3) TEST get_capitalized_word_count()
get_capitalized_word_count(test_feature_df, test_df)

result_df['number_of_capital_words'] = [0, 0]

pd.testing.assert_frame_equal(get_capitalized_word_count(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 4) TEST get_small_word_count()

result_df['number_of_small_words'] = [4, 5]
get_small_word_count(test_feature_df, test_df)

pd.testing.assert_frame_equal(get_small_word_count(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 5) TEST get_number_of_alphabets()

get_number_of_alphabets(test_feature_df, test_df)

result_df['number_of_alphabets'] = [10, 16]

pd.testing.assert_frame_equal(get_number_of_alphabets(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 6) get_number_of_digit_count()

get_number_of_digit_count(test_feature_df, test_df)

result_df['number_of_digits'] = [0, 0]

pd.testing.assert_frame_equal(get_number_of_digit_count(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 7) TEST get_number_of_words()

get_number_of_words(test_feature_df, test_df)

result_df['number_of_words'] = [4, 5]

pd.testing.assert_frame_equal(get_number_of_words(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)

# 8) TEST get_number_of_whitespaces()

get_number_of_whitespaces(test_feature_df, test_df)

result_df['number_of_white_spaces'] = [4, 4]

pd.testing.assert_frame_equal(get_number_of_whitespaces(test_feature_df, test_df), result_df, check_names=False,
                              check_like=True)
