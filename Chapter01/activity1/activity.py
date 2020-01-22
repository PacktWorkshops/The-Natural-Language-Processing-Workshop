# @author Muzaffar

from nltk import download

download('stopwords')
download('wordnet')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from autocorrect import spell
from nltk import stem, pos_tag
from nltk.tokenize import sent_tokenize
import string

"""
In this code we will load a text file and apply all
the basic nlp tasks on the text.
"""
# load the text file into variable called sentence
sentence = open("../../data_ch1/file.txt", 'r').read()

# tokenize the sentence
words = word_tokenize(sentence)
# print the first 20 words.
print(words[0:20])

# The code block below will correct the spelling
# of the misspelled words
corrected_sentence = ""
corrected_word_list = []
for wd in words:
    if wd not in string.punctuation:
        wd_c = spell(wd)
        if wd_c != wd:
            print(wd+" has been corrected to: "+wd_c)
            corrected_sentence = corrected_sentence+" "+wd_c
            corrected_word_list.append(wd_c)
        else:
            corrected_sentence = corrected_sentence+" "+wd
            corrected_word_list.append(wd)
    else:
        corrected_sentence = corrected_sentence + wd
        corrected_word_list.append(wd)
print(corrected_sentence)

# print all the list of corrected words
print(corrected_word_list[:20])

# Get pos tag of corrected word list
print(pos_tag(corrected_word_list))

# The code block below will remove the stop words
# from the text.
stop_words = stopwords.words('english')
corrected_word_list_without_stopwords = []
for wd in corrected_word_list:
    if wd not in stop_words:
        corrected_word_list_without_stopwords.append(wd)
print(corrected_word_list_without_stopwords[:20])


# The code block below will get stem of word list
# from above code.

stemmer = stem.PorterStemmer()
corrected_word_list_without_stopwords_stemmed = []
for wd in corrected_word_list_without_stopwords:
    corrected_word_list_without_stopwords_stemmed.append(stemmer.stem(wd))
print(corrected_word_list_without_stopwords_stemmed[:20])

# The code block below will get lemmas of word list
# from above code.
lemmatizer = WordNetLemmatizer()
corrected_word_list_without_stopwords_lemmatized = []
for wd in corrected_word_list_without_stopwords:
    corrected_word_list_without_stopwords_lemmatized.append(lemmatizer.lemmatize(wd))
print(corrected_word_list_without_stopwords_lemmatized[:20])

# The code line below will split the text into sentence.
print(sent_tokenize(corrected_sentence))






