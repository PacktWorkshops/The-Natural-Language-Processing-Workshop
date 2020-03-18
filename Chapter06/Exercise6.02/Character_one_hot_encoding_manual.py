import unittest


def onehot_word(word):
    lookup = {v[1]: v[0] for v in enumerate(set(word))}
    word_vector = []
    for c in word:
        one_hot_vector = [0] * len(lookup)
        one_hot_vector[lookup[c]] = 1
        word_vector.append(one_hot_vector)
    return word_vector


onehot_vector = onehot_word('data')
print(onehot_vector)


class TestMethods(unittest.TestCase):

    def test_get_lemmas(self):
        """
        Adding count equal test because can not guarntee the order of 0 and 1's in inside
        indvidual array
        :return: None
        """
        result = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        self.assertCountEqual(onehot_word('this'), result)


if __name__ == '__main__':
    unittest.main()
