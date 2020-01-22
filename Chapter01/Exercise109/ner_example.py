# @author Muzaffar

from nltk import download
from nltk import word_tokenize
from nltk import ne_chunk, pos_tag
# load maxent chunker model
download('maxent_ne_chunker')
download('words')


sentence = "We are reading a book published by Packt which is based out of Birmingham."

# The below line of code will identifies the named entities from the sentence
i = ne_chunk(pos_tag(word_tokenize(sentence)), binary=True)
print([a for a in i if len(a)==1])

