sentence = 'The quick brown fox jumps over the lazy dog'
print(sentence)

# Find if 'quick' is in sentence
print('quick' in sentence)
# Expected output:- True

# Find index of 'fox'
print(sentence.index('fox'))
# Expected output:- 16

# Find out the rank of the word 'lazy',
# split() method splits a sentence into words.
print(sentence.split().index('lazy'))
# Expected output:- 7

# Print the third word of sentence (word at index 2)
print(sentence.split()[2])
# Expected output:- brown

# Print the third word in reverse order
print(sentence.split()[2][::-1])
# Expected output:- nworb

# concat first and third word of a given senetence
words = sentence.split()  # split sentence into words
first_word = words[0]  # Get the first word
last_word = words[-1]  # Get the last word
concat_word = first_word + last_word
print(concat_word)
# Expected output:- Thedog

# Printing words at even positions
print([words[i] for i in range(len(words)) if i % 2 == 0])
# Expected output:- ['The', 'brown', 'jumps', 'the', 'dog']

# To print the last 3 words of sentence.
print(sentence[-3:])
# Expected output:- dog

# To print the text in reverse order, use the following code.
print(sentence[::-1])
# Expected output:- god yzal eht revo spmuj xof nworb kciuq ehT

# print each word of the given text in reverse order, maintaining their sequence
print(' '.join([word[::-1] for word in words]))
# Expected output:- ehT kciuq nworb xof spmuj revo eht yzal god
