import unittest

import numpy as np
import zipfile

GLOVE_DIR = '../data/'
GLOVE_ZIP = GLOVE_DIR + 'glove.6B.50d.txt.zip'
print(GLOVE_ZIP)

zip_ref = zipfile.ZipFile(GLOVE_ZIP, 'r')
zip_ref.extractall(GLOVE_DIR)
zip_ref.close()


def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open(fn, 'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model


glove_vectors = load_glove_vectors(GLOVE_DIR + 'glove.6B.50d.txt')

print(glove_vectors)

print(glove_vectors["dog"])

print(glove_vectors["cat"])


# the below method is just for test case and is not in exercise

def get_vector(word):
    return glove_vectors[word]


import matplotlib.pyplot as plt


def to_vector(glove_vectors, word):
    vector = glove_vectors.get(word.lower())
    if vector is None:
        vector = [0] * 50
    return vector


def to_image(vector, word=''):
    fig, ax = plt.subplots(1, 1)
    ax.tick_params(axis='both', which='both',
                   left=False,
                   bottom=False,
                   top=False,
                   labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.bar(range(len(vector)), vector, 0.5)
    ax.text(s=word, x=1, y=vector.max() + 0.5)
    return vector


man = to_image(to_vector(glove_vectors, "man"))
woman = to_image(to_vector(glove_vectors, "woman"))
king = to_image(to_vector(glove_vectors, "king"))
queen = to_image(to_vector(glove_vectors, "queen"))
diff = to_image(king - man + woman - queen)
nd = to_image(king - man + woman)


class TestMethods(unittest.TestCase):

    def test_get_lemmas(self):
        result = 0.003364140000000012
        self.assertEqual(get_vector("focus").mean(), result)


if __name__ == '__main__':
    unittest.main()
