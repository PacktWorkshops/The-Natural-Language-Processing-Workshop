import pandas as pd
from string import punctuation
import nltk

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


def get_pos_occurrence_freq(data, tag_list):
    # Get list of sentences in text_list
    text_list = data.text

    # create empty dataframe
    feature_df = pd.DataFrame(tag_list)
    for text_line in text_list:
        # get pos tags of each word.
        pos_tags = [j for i, j in pos_tag(word_tokenize(text_line))]

        # create a dict of pos tags and their frequency in given sentence.
        row = dict(Counter(pos_tags))
        feature_df = feature_df.append(row, ignore_index=True)
    feature_df.fillna(0, inplace=True)
    return feature_df


def add_punctuation_count(feature_df, data):
    # The below code line will find the intersection of set
    # of punctuations in text and punctuation set
    # imported from string module of python and find the length of
    # intersection set in each row and add it to column `num_of_unique_punctuations`
    # of data frame.

    feature_df['num_of_unique_punctuations'] = data['text']. \
        apply(lambda x: len(set(x).intersection(set(punctuation))))
    return feature_df


#
def get_capitalized_word_count(feature_df, data):
    # The below code line will tokenize text in every row and
    # create a set of only capital words, then find the length of
    # this set and add it to the column `number_of_capital_words`
    # of dataframe.

    feature_df['number_of_capital_words'] = data['text']. \
        apply(lambda x: len([word for word in word_tokenize(str(x)) if word[0].isupper()]))
    return feature_df


def get_small_word_count(feature_df, data):
    # The below code line will tokenize text in every row and
    # create a set of only small words, then find the length of
    # this set and add it to the column `number_of_small_words`
    # of dataframe.

    feature_df['number_of_small_words'] = data['text']. \
        apply(lambda x: len([word for word in word_tokenize(str(x)) if word[0].islower()]))
    return feature_df


def get_number_of_alphabets(feature_df, data):
    # The below code line will generate list of alphabets
    # in each row and add the length of that list into
    # the columns `number_of_alphabets`

    feature_df['number_of_alphabets'] = data['text']. \
        apply(lambda x: len([ch for ch in str(x) if ch.isalpha()]))
    return feature_df


def get_number_of_digit_count(feature_df, data):
    # The below code line will generate list of digts
    # in each row and add the length of that list into
    # the columns `number_of_digits`

    feature_df['number_of_digits'] = data['text']. \
        apply(lambda x: len([ch for ch in str(x) if ch.isdigit()]))
    return feature_df


def get_number_of_words(feature_df, data):
    # The below code line will break the text line in a list of
    # words in each row and add the count of that list into
    # the columns `number_of_digits`

    feature_df['number_of_words'] = data['text'].apply(lambda x
                                                       : len(word_tokenize(str(x))))

    return feature_df


def get_number_of_whitespaces(feature_df, data):
    # The below code line will generate list of white spaces
    # in each row and add the length of that list into
    # the columns `number_of_white_spaces`

    feature_df['number_of_white_spaces'] = data['text']. \
        apply(lambda x: len([ch for ch in str(x) if ch.isspace()]))

    return feature_df


data = pd.read_csv('../../data/data.csv', header=0)

feature_df = get_pos_occurrence_freq(data, tag_list)
print(feature_df.head())

feature_df = add_punctuation_count(feature_df, data)
print(feature_df['num_of_unique_punctuations'].head())

feature_df = get_capitalized_word_count(feature_df, data)
print(feature_df['number_of_capital_words'].head())

feature_df = get_small_word_count(feature_df, data)
print(feature_df['number_of_small_words'].head())

feature_df = get_number_of_alphabets(feature_df, data)
print(feature_df['number_of_alphabets'].head())

feature_df = get_number_of_digit_count(feature_df, data)
print(feature_df['number_of_digits'].head())

feature_df = get_number_of_words(feature_df, data)
print(feature_df['number_of_words'].head())


feature_df = get_number_of_whitespaces(feature_df, data)
print(feature_df['number_of_white_spaces'].head())
