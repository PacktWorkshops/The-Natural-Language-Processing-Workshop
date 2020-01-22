# @author Muzaffar

from nltk import download
download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

# create an object of WordNetLemmatizer class
lemmatizer = WordNetLemmatizer()

# The below code line will return product
print(lemmatizer.lemmatize('products'))

print(lemmatizer.lemmatize('coming'))

print(lemmatizer.lemmatize('battle'))
