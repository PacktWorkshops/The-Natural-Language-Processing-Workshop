import operator

from nltk.tokenize import WhitespaceTokenizer
from nltk import download, stem

# The below statement will download the stop word list
# 'nltk_data/corpora/stopwords/' at home directory of your computer
download('stopwords')
from nltk.corpus import stopwords


def load_file(file_path):
    # load the new article
    news = ''.join([line for line in open(file_path)])
    return news


def to_lower_case(text):
    return text.lower()


def tokenize_text(text):
    wht = WhitespaceTokenizer()
    return wht.tokenize(text=text)


def remove_stop_words(token_list):
    stop_words = stopwords.words('english')
    return [word for word in token_list if word not in stop_words]


def get_stems(token_list):
    stemmer = stem.PorterStemmer()
    return [stemmer.stem(word) for word in token_list]


def get_freq(stems):
    freq_dict = {}
    for t in stems:
        freq_dict[t.strip()] = freq_dict.get(t.strip(), 0) + 1
    return freq_dict


def get_top_n_words(freq_dict, n):
    sorted_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict][:n]


if __name__ == '__main__':
    path = "../data/news_article.txt"
    news_article = load_file(path)
    lower_case_news_art = to_lower_case(text=news_article)
    tokens = tokenize_text(lower_case_news_art)
    removed_tokens = remove_stop_words(tokens)
    stems = get_stems(removed_tokens)
    freq_dict = get_freq(stems)
    top_keywords = get_top_n_words(freq_dict, 6)
    print(top_keywords)
