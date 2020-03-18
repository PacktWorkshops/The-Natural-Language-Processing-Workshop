import pandas as pd
from IPython.core.display import display
from textblob import TextBlob

df = pd.DataFrame([['The interim budget for 2019 will be announced on 1st February.'],
                   ['Do you know how much expectation the middle-class working population is having from this budget?'],
                   ['February is the shortest month in a year.'], ['This financial year will end on 31st March.']])
df.columns = ['text']
display(df.head())


def add_num_words(df):
    df['number_of_words'] = df['text'].apply(lambda x: len(TextBlob(str(x)).words))
    return df


display(add_num_words(df)['number_of_words'])


def is_present(wh_words, df):
    # The below line of code will find the intersection between set of tokens of
    #  every sentence and the wh_words and will return true if the length of intersection
    #  set is non-zero.
    df['is_wh_words_present'] = df['text'].apply(
        lambda x: True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)
    return df


wh_words = {'why', 'who', 'which', 'what', 'where', 'when', 'how'}

display(is_present(wh_words, df)['is_wh_words_present'])


def get_language(df):
    df['language'] = df['text'].apply(lambda x: TextBlob(str(x)).detect_language())
    return df


display(get_language(df)['language'])

# Below code is for test case
# Create a test data frame
test_df = pd.DataFrame([{"text": "this is  a cat"}, {"text": "why are you so happy"}])

# 1) TEST  add_num_words()

# create a expected result data frame 
result_df = test_df.copy(deep=True)
result_df['number_of_words'] = [4, 5]

# Assert equality of the results and expected data frames

pd.testing.assert_frame_equal(add_num_words(test_df), result_df, check_names=False)

# 2) TEST  is_present()
words_to_check = ['why', 'where', 'who']
result_df['is_wh_words_present'] = [False, True]
pd.testing.assert_frame_equal(is_present(words_to_check, test_df), result_df, check_names=False)

# 3) TEST get_language()

result_df['language'] = ['en', 'en']
pd.testing.assert_frame_equal(get_language(test_df), result_df, check_names=False)
