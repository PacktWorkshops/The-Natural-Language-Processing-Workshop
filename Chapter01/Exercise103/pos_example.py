# @author Muzaffar

from nltk import word_tokenize, pos_tag

"""
This is an example code for generating pos tags of given text.
"""

# The below method will split text sentence into tokens.
words = word_tokenize("I am reading NLP Fundamentals")
print(words)

# The below method assign tag to every token.
# This will return a list of tuples, each tuple containing word
# and the pos tag of the word
pos_tags = pos_tag(words)

print(pos_tags)
