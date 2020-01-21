"""
This code will help in replacing the substring with another substring
 from the given string
"""

sentence = "I visited US from UK on 22-10-18"

# This below line of code wil replace US with United States
# and UK with United Kingdom.
normalized_sentence = sentence.replace("US", "United States") \
    .replace("UK", "United Kingdom").replace("-18", "-2018")

print(normalized_sentence)

