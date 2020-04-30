import unittest

from keras.preprocessing.text import Tokenizer
import numpy as np

char_tokenizer = Tokenizer(char_level=True)
text = 'The quick brown fox jumped over the lazy dog'
char_tokenizer.fit_on_texts(text)
seq = char_tokenizer.texts_to_sequences(text)
print(seq)

char_tokenizer.sequences_to_texts(seq)
char_vectors = char_tokenizer.texts_to_matrix(text)
print(char_vectors)

print(char_vectors.shape)

print(char_vectors[0])

np.argmax(char_vectors[0])
print(char_tokenizer.index_word)

print(char_tokenizer.word_index)

print(char_tokenizer.index_word[np.argmax(char_vectors[0])])


# ### Adding this method below just for test case and is not present in the exercise

def get_one_hot_vector(text_, word_index):
    char_tokenizer.fit_on_texts(text_)
    char_vectors = char_tokenizer.texts_to_matrix(text_)
    return list(char_vectors[word_index])


print(get_one_hot_vector("This is a cat", 0))


class TestMethods(unittest.TestCase):

    def test_get_lemmas(self):
        result = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertEqual(get_one_hot_vector("This is a cat", 0), result)


if __name__ == '__main__':
    unittest.main()
