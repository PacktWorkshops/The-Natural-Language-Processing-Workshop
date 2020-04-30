"""
This python code contains basic text analytic functionaries.
The unit test doc is written at the begining of every string.
And the unit testing is done at the last cell of this notebook
"""


# @author Muzaffar


def find_word(word, sentence):
    '''
    This is a test case:
    >>> find_word('quick',  'The quick brown fox jumps over the lazy dog')
    True
    '''
    return word in sentence


def get_index(word, text):
    '''
    This is a test case:
    >>> get_index('fox',  'The quick brown fox jumps over the lazy dog')
    16
    '''
    return text.index(word)


def get_word(text, rank):
    '''
    >>> get_word('The quick brown fox jumps over the lazy dog',2)
    'brown'
    '''
    return text.split()[rank]


def concat_words(text):
    """
    This method will concat first and last
    words of given text

    The below line is for unit tests
    >>> concat_words('This is for unit testing')
    'Thistesting'

    """
    words = text.split()
    first_word = words[0]
    last_word = words[len(words) - 1]
    return first_word + last_word


def get_even_position_words(text):
    """
    The below line is for unit tests
    >>> get_even_position_words('This is for unit testing')
    ['This', 'for', 'testing']
    """
    words = text.split()
    return [words[i] for i in range(len(words)) if i % 2 == 0]


def get_last_n_letters(text, n):
    """
    The below line is for unit tests
    >>> get_last_n_letters('This is for unit testing',4)
    'ting'
    """
    return text[-n:]


def get_reverse(text):
    """
    The below line is for unit tests
    >>> get_reverse('This is for unit testing')
    'gnitset tinu rof si sihT'
    """

    return text[::-1]


def get_word_reverse(text):
    """
     >>> get_word_reverse('This is for unit testing')
     'sihT si rof tinu gnitset'
    """
    words = text.split()
    return ' '.join([word[::-1] for word in words])


sentence = 'The quick brown fox jumps over the lazy dog'
print(find_word('quick', sentence))
print(get_index('fox', sentence))
print(get_index('lazy', sentence.split()))
print(get_word(sentence, 2))
print(get_word(sentence, 2)[::-1])
print(get_even_position_words(sentence))
print(get_word_reverse(sentence))
print(get_last_n_letters(sentence, 3))
print(get_reverse(sentence))
print(concat_words(sentence))

import doctest
doctest.testmod(verbose=True)
