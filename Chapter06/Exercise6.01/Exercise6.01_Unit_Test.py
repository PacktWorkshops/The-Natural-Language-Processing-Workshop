#!/usr/bin/env python
# coding: utf-8
import unittest
from pathlib import Path
import string
import re
import numpy as np
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')

data = Path('../data')
novel_lines_file = data / 'novel_lines.txt'

with novel_lines_file.open() as f:
    novel_lines_raw = f.read()

print(novel_lines_raw)
alpha_characters = str.maketrans('', '', string.punctuation)


def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r'\n', '*** ', text)
    text = text.translate(alpha_characters)
    text = re.sub(r' +', ' ', text)
    return text.strip().split(' ')


novel_lines = clean_tokenize(novel_lines_raw)

print(novel_lines)

novel_lines_array = np.array([novel_lines])
novel_lines_array = novel_lines_array.reshape(-1, 1)
print(novel_lines_array.shape)

labelEncoder = preprocessing.LabelEncoder()
novel_lines_labels = labelEncoder.fit_transform(novel_lines_array)

wordOneHotEncoder = preprocessing.OneHotEncoder()

line_onehot = wordOneHotEncoder.fit_transform(novel_lines_labels.reshape(-1, 1))

print(novel_lines_labels)

print(line_onehot)

print(line_onehot.toarray())


class TestMethods(unittest.TestCase):

    def test_clean_tokenize(self):
        result = ['this', 'is', 'a', 'cat']
        self.assertEqual(clean_tokenize("This ' is a \  cat"), result)


if __name__ == '__main__':
    unittest.main()
