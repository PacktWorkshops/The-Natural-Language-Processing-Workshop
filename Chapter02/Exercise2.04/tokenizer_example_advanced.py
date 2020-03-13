#!/usr/bin/env python
# coding: utf-8

# 
# # This notebook contains advanced tokemizer examples

# In[ ]:


from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import WordPunctTokenizer


# In[4]:


sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'


# In[5]:


def tokenize_with_tweet_tokenizer(text):
    """
    >>> tokenize_with_tweet_tokenizer('Awesome airshow! @india_official  @indian_army #India #70thRepublic_Day.')
    ['Awesome', 'airshow', '!', '@india_official', '@indian_army', '#India', '#70thRepublic_Day', '.']
    """
    tweet_tokenizer = TweetTokenizer() # Here will create an object of tweetTokenizer
    return tweet_tokenizer.tokenize(text) # Then we will call the tokenize 
                                       # method oftweetTokenizer which will return token list of sentence.


# In[6]:


tokenize_with_tweet_tokenizer(sentence)


# In[11]:


def tokenize_with_mwe(text):
    """
    >>> tokenize_with_mwe('Sunil tweeted, "Witnessing 70th Republic Day of India')
    ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic_Day', 'of', 'India']
    
    """
    mwe_tokenizer = MWETokenizer([('Republic', 'Day')])
    mwe_tokenizer.add_mwe(('Indian', 'Army'))
    return mwe_tokenizer.tokenize(text.split())


# In[12]:


tokenize_with_mwe(sentence)


# In[13]:


tokenize_with_mwe(sentence.replace('!',''))


# In[22]:


def tokenize_with_regex_tokenizer(text):
    '''
    Test case string
    >>> tokenize_with_regex_tokenizer('Sunil tweeted, "Witnessing 70th Republic Day of India')
    ['Sunil', 'tweeted', ',', '"Witnessing', '70th', 'Republic', 'Day', 'of', 'India']
    '''
    reg_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    return reg_tokenizer.tokenize(text)


# In[20]:


tokenize_with_regex_tokenizer(sentence)


# In[32]:


def tokenize_with_wst(text):
    '''
    >>> tokenize_with_wst('Sunil tweeted, "Witnessing 70th Republic Day of India')
    ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic', 'Day', 'of', 'India']
    '''
    wh_tokenizer = WhitespaceTokenizer()
    return wh_tokenizer.tokenize(text)


# In[33]:


tokenize_with_wst(sentence)


# In[36]:


def tokenize_with_wordpunct_tokenizer(text):
    '''
    >>> tokenize_with_wordpunct_tokenizer(' For more photos ping me sunil@photoking.com :)')
    ['For', 'more', 'photos', 'ping', 'me', 'sunil', '@', 'photoking', '.', 'com', ':)']
    '''
    wp_tokenizer = WordPunctTokenizer()
    return wp_tokenizer.tokenize(text)


# In[37]:


tokenize_with_wordpunct_tokenizer(sentence)


# In[38]:


import doctest

doctest.testmod(verbose=True)


# In[ ]:





# In[ ]:




