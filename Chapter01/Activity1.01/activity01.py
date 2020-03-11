#!/usr/bin/env python
# coding: utf-8

# # ACTIVITY 1
# Read the file at data_ch1 and perform the basic NLP tasks on the text like sentence tokenizatio, word topkenization, stemming, lemmatization, splell checking, etc

# In[ ]:


from nltk import download
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from autocorrect import Speller
from autocorrect import spell
from nltk.wsd import lesk
from nltk.tokenize import sent_tokenize
from nltk import stem, pos_tag
import string


# In[ ]:


sentence = open("../data/file.txt", 'r').read()


# In[ ]:


words = word_tokenize(sentence)


# In[ ]:


print(words[0:20])


# In[ ]:


spell = Speller(lang='en')

def correct_sentence(words):
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
    return corrected_sentence, corrected_word_list


# In[ ]:


corrected_sentence, corrected_word_list = correct_sentence(words)


# In[ ]:


corrected_sentence


# In[ ]:


print(corrected_word_list[0:20])


# In[ ]:


print(pos_tag(corrected_word_list))


# In[ ]:


stop_words = stopwords.words('english')
def remove_stop_words(word_list):
    corrected_word_list_without_stopwords = []
    for wd in word_list:
        if wd not in stop_words:
            corrected_word_list_without_stopwords.append(wd)
    return corrected_word_list_without_stopwords

corrected_word_list_without_stopwords = remove_stop_words(corrected_word_list)
corrected_word_list_without_stopwords[:20]


# In[ ]:


stemmer = stem.PorterStemmer()
def get_stems(word_list):
    corrected_word_list_without_stopwords_stemmed = []
    for wd in word_list:
        corrected_word_list_without_stopwords_stemmed.append(stemmer.stem(wd))
    return corrected_word_list_without_stopwords_stemmed

corrected_word_list_without_stopwords_stemmed = get_stems(corrected_word_list_without_stopwords)
corrected_word_list_without_stopwords_stemmed[:20]


# In[ ]:


lemmatizer = WordNetLemmatizer()
def get_lemma(word_list):
    corrected_word_list_without_stopwords_lemmatized = []
    for wd in word_list:
        corrected_word_list_without_stopwords_lemmatized.append(lemmatizer.lemmatize(wd))
    return corrected_word_list_without_stopwords_lemmatized
corrected_word_list_without_stopwords_lemmatized =  get_lemma(corrected_word_list_without_stopwords_stemmed)
corrected_word_list_without_stopwords_lemmatized[:20]


# In[ ]:


print(sent_tokenize(corrected_sentence))


# In[ ]:




