#  Text Visualization
import unittest

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import re
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

text = open('../data/text_corpus.txt', 'r', encoding="utf8").read()

print(text[:1040])


def lemmatize_and_clean(text):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    cleaned_lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in
                                 word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', text))]
    return cleaned_lemmatized_tokens


Counter(lemmatize_and_clean(text)).most_common(50)
stopwords = set(STOPWORDS)
cleaned_text = ' '.join(lemmatize_and_clean(text))
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      max_words=50,
                      stopwords=stopwords,
                      min_font_size=10).generate(cleaned_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


class TestMethods(unittest.TestCase):

    def test_lemmatize_and_clean(self):
        result = ['this', 'string', 'is', 'for', 'testing', 'and', 'creating', 'use', 'case']
        self.assertEqual(lemmatize_and_clean("This String is for testing and creating use cases"), result)


if __name__ == '__main__':
    unittest.main()
