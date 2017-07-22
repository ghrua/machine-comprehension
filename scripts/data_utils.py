import os
import re
import time
import argparse
from tensorflow import gfile
from collections import Counter
from nltk.corpus import stopwords
from os.path import dirname, abspath
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'@?\w+')


# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"(^| )\d+")

_ENTITY = "@entity"
_BAR = "_BAR"
_UNK = "_UNK"
BAR_ID = 0
UNK_ID = 1
_START_VOCAB = [_BAR, _UNK]

tokenizer = RegexpTokenizer(r'@?\w+')
cachedStopWords = stopwords.words("english")

def read_data()

if __name__ == "__main__":
    pass
