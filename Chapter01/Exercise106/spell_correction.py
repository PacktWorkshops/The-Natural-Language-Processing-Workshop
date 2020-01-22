# @author Muzaffar

from autocorrect import spell
from nltk import word_tokenize


# The below code line will return correct spell 'Natural'

print(spell('Natureal'))

# Tokenize a given sentence
sentence = word_tokenize("Ntural Luanguage Processin deals with the art"
                         " of extracting insightes from Natural Languaes")

print(sentence)

# recreate the sentence with correct spell of each word
sentence_corrected = ' '.join([spell(word) for word in sentence])
print(sentence_corrected)
