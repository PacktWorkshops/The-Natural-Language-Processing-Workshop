import nltk
from nltk import word_tokenize
# load maxent chunker model
nltk.download('maxent_ne_chunker')
nltk.download('words')


sentence = "We are reading a book published by Packt which is based out of Birmingham."

# The below line of code will identifies the named entities from the sentence
i = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=True)
print([a for a in i if len(a)==1])

