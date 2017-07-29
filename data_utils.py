# Modification of https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/rnn/translate/data_utils.py
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
import os
import re
import sys
import argparse

import numpy as np
from tqdm import *
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
from tensorflow.python.platform import gfile

# find the words
_WORD = re.compile("@?\w+")
# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

_ENTITY = "@entity"
_BAR = "_BAR"
_UNK = "_UNK"
BAR_ID = 0
UNK_ID = 1
_START_VOCAB = [_BAR, _UNK]

cachedStopWords = stopwords.words("english")


def tokenizer(text):
    return _WORD.findall(text)


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = tokenizer(sentence)
    return [w for w in words if w not in stopwords.words("english")]


def create_vocabulary(vocabulary_path, counter, max_vocabulary_size):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("[{}] Creating vocabulary".format(datetime.now()))
        vocab, _ = zip(*counter.most_common(max_vocabulary_size))
        np.save(vocabulary_path, vocab)
        print("[{}] Vocabulary is saved at: {}".format(datetime.now(), vocabulary_path))


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        vocab = np.load(vocabulary_path)
        s2i = dict(zip(vocab, range(2, 2 + len(vocab))))
        return s2i, list(vocab)
    else:
        raise ValueError("Vocabulary file %s not found." % vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, vocab):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Return:
      counetr: a Counter() instance get from the data
    """
    with gfile.GFile(data_path, mode="r") as data_file:
        results = []
        data = data_file.read()
        seq = data.split("\n\n")
        for it in range(1, 3):
            results.append(sentence_to_token_ids(seq[it], vocab))
        return tuple(results)


def build_counter_from_context(context_fname):
    """
    Args:
        context_fname: read the context from this file
    Return:
        counter: an instance of collection.Counter(), which stores the words and its frequency in the data
    """
    print("[{}] Building the Counter of data".format(datetime.now()))
    counter = Counter()
    print("[{}] Building Counter from existed context file: {}".format(datetime.now(), context_fname))
    with open(context_fname) as context:
        for line in context:
            counter.update(basic_tokenizer(line))
    print("Done...")
    return counter


def get_the_counter(dir_name, context_fname):
    """
    Args:
        dir_name: the directory data located at
        context_fname: write the context to this file
    Return:
        counter: an instance of collection.Counter(), which stores the words and its frequency in the data
    """
    counter = Counter()
    fout = open(context_fname, 'w')
    for fname in tqdm(gfile.Glob(os.path.join(dir_name, "*.question"))):
        with open(fname) as f:
            context = ""
            try:
                lines = f.read().split("\n\n")
                context += lines[1] + " "
                context += lines[2] + " "
                context += lines[4].replace(":", " ") + " "
            except Exception:
                print(" [!] Error occurred for {}".format(fname))
                continue
            fout.write(context)
            counter.update(basic_tokenizer(context))
    print(" [{}] Context is writing at: {}".format(datetime.now(), context_fname))
    fout.close()
    return counter


def questions_to_token_ids(files, vocab):
    ret = []
    for it in tqdm(files):
        ret.append(data_to_token_ids(it, vocab))
    return tuple(ret)


def prepare_data(data_dir, output_dir, dataset_name, vocab_size):
    context_fname = os.path.join(output_dir, '%s.context' % dataset_name)
    vocab_fname = os.path.join(output_dir, '%s.vocab%s' % (dataset_name, str(vocab_size)))

    if gfile.Exists(context_fname):
        counter = build_counter_from_context(context_fname)
    else:
        counter = get_the_counter(data_dir, context_fname)
    print(" [*] Skip combining all contexts")
    print(len(counter))

    if not os.path.exists(vocab_fname):
        print(" [*] Create vocab from %s to %s ..." % (context_fname, vocab_fname))
        create_vocabulary(vocab_fname, counter, vocab_size)
    else:
        print(" [*] Skip creating vocab")


def load_vocab(data_dir, dataset_name, vocab_size):
    vocab_fname = os.path.join(data_dir, dataset_name, "%s.vocab%s" % (dataset_name, vocab_size))
    print(" [*] Loading vocab from %s ..." % vocab_fname)
    return initialize_vocabulary(vocab_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", help="语料所在目录")
    parser.add_argument("-o", "--out_dir", help="输出目录")
    parser.add_argument("-s", "--set", help="数据集名称")
    parser.add_argument("-v", "--vocab_size", help="字典大小", type=int)
    args = parser.parse_args()
    prepare_data(args.data_dir, args.out_dir, args.set, args.vocab_size)
