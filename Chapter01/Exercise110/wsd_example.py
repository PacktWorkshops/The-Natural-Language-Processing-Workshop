# @author Muzaffar

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

sentence1 = "Keep your savings in the bank"
sentence2 = "It's so risky to drive over the banks of the road"

# The below line of code will print Synset('savings_bank.n.02')
# it is one of the definition sof word 'bank'
print(lesk(word_tokenize(sentence1), 'bank'))

print(lesk(word_tokenize(sentence2), 'bank'))

# The below code will print all the definitions of word bank
wn.synsets('bank')
for ss in wn.synsets('bank'):
    print(ss, ss.definition())
