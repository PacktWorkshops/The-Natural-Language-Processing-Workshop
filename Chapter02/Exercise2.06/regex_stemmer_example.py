import unittest

from nltk.stem import RegexpStemmer

# It is better to create object of RegexpStemmer here.
#  out side method
regex_stemmer = RegexpStemmer('ing$', min=4)  # creating an object of RegexpStemmer,


# any string ending with the given
# regex ‘ing$’ will be removed.
def get_stems(text):
    # The below code line will convert every word into its stem using regex stemmer and then join them with space.
    return ' '.join([regex_stemmer.stem(wd) for wd in text.split()])


sentence = "I love playing football"
print(get_stems(sentence))


class TestMethods(unittest.TestCase):

    def test_get_stems(self):
        result = 'I was go there'
        self.assertEqual(get_stems('I was going there'), result)


if __name__ == '__main__':
    unittest.main()
