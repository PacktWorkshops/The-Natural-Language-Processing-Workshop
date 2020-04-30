# # This notebook shows how to generate wordcloud on a given corpus using wordcloud library

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200


# In[ ]:


def get_data(n):
    newsgroups_data_sample = fetch_20newsgroups(subset='train')
    text = str(newsgroups_data_sample['data'][:n])
    return text


def load_stop_words():
    other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI', "n'"]
    stop_words = stopwords.words('english')
    stop_words.extend(other_stopwords_to_remove)
    stop_words = set(stop_words)
    return stop_words


def generate_word_cloud(text, stopwords):
    """
    
    This method generates word cloud object
    with given corpus, stop words and dimensions
    """

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=200,
                          stopwords=stopwords,
                          min_font_size=10).generate(text)
    return wordcloud


text = get_data(10000)
stop_words = load_stop_words()
wordcloud = generate_word_cloud(text, stop_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
