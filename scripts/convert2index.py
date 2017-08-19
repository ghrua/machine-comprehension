import sys
import random
import numpy as np
from functools import partial
from sklearn.externals import joblib
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import tensorflow as tf
import pickle as pk
from os.path import abspath, dirname, join, exists, basename
from os import makedirs

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils import padding
from data_utils import questions_to_token_ids, load_vocab


class Config:
    vocab_size = 100002
    embed_size = 200
    state_size = 256
    depth = 2
    lr = 5E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 200


def gen_file(it, dump_dir, vocabulary):
    p, q, a = zip(*questions_to_token_ids(it, vocabulary))
    inputs = list(p[0]) + [0] + list(q[0])
    length = len(inputs)
    dump_obj = [inputs, length, a[0][0]]
    if not exists(dump_dir):
        makedirs(dump_dir)
    joblib.dump(dump_obj, join(dump_dir, basename(it)))


def load(root_dir):
    test_set = tf.gfile.Glob(join(root_dir, "test", "*.question"))
    training_set = tf.gfile.Glob(join(root_dir, "training", "*.question"))
    validation_set = tf.gfile.Glob(join(root_dir, "validation", "*.question"))
    vocabulary, _ = load_vocab(root_dir, str(Config.vocab_size - 2))
    return training_set, test_set, validation_set, vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="training, test, validation, vocabulary 所在")
    parser.add_argument("-d", "--dump", help="转化后文件所在的目录")
    args = parser.parse_args()
    training_set, test_set, validation_set, vocabulary = load(args.root)
    pool = Pool(4)
    pool.map(partial(gen_file, dump_dir=join(args.dump, 'training'), vocabulary=vocabulary), training_set)
    pool.map(partial(gen_file, dump_dir=join(args.dump, 'test'), vocabulary=vocabulary), test_set)
    pool.map(partial(gen_file, dump_dir=join(args.dump, 'validation'), vocabulary=vocabulary), validation_set)

