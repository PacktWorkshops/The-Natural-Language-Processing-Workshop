from nltk.wsd import lesk
from nltk import word_tokenize


def get_synset(sentence, word):
    return lesk(word_tokenize(sentence), word)


sentence1 = "Keep your savings in the bank"
sentence2 = "It's so risky to drive over the banks of the road"
print(get_synset(sentence1, 'bank'))
print(get_synset(sentence2, 'bank'))
