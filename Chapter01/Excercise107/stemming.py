# @author Muzaffar

from nltk import stem

# create a object of porterStemmer class
stemmer = stem.PorterStemmer()

# Will return base form 'product' of the word.
print(stemmer.stem("production"))

print(stemmer.stem("coming"))

print(stemmer.stem("firing"))

print(stemmer.stem("battling"))