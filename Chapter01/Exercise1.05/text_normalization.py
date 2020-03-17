import unittest


def normalize(text):
    return text.replace("US", "United States").replace("UK", "United Kingdom").replace("-18", "-2018")


class TestMethods(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize('US and UK are two superpowers'),
                         'United States and United Kingdom are two superpowers')


if __name__ == '__main__':
    unittest.main()
