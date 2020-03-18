import unittest

from textblob import TextBlob


def translate(text, from_l, to_l):
    en_blob = TextBlob(text)
    return en_blob.translate(from_lang=from_l, to=to_l)


translate(text='muy bien', from_l='es', to_l='en')
print(translate('Hello', 'en', 'es'))


class TestMethods(unittest.TestCase):

    def test_get_lemmas(self):
        result = TextBlob("Hola")
        self.assertEqual(translate('Hello', 'en', 'es'), result)


if __name__ == '__main__':
    unittest.main()
