import sys
import random
import numpy as np
import argparse
import tensorflow as tf
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils import padding
from data_utils import questions_to_token_ids, load_vocab


class Config:
    vocab_size = 100002  # 这里有问题
    embed_size = 256
    state_size = 128
    depth = 2
    lr = 1e-4
    batch_size = 32
    num_sampled = 128
    n_epoch = 10
    train_steps = 10


class DataHelperTest:
    def __init__(self, root_dir, data_set_name):
        self.data_set = tf.gfile.Glob(join(root_dir, data_set_name, "*.question"))
        self.vocabulary, _ = load_vocab(root_dir, str(Config.vocab_size-2))

    def gen_file(self):
        samples = random.sample(list(self.data_set), 1)
        gen_tuples = questions_to_token_ids(samples, self.vocabulary)
        print(gen_tuples)
        print("shape: {}".format(np.asarray(gen_tuples).shape))
        inp = input("==========\n")
        p, q, a = zip(*gen_tuples)
        print("p: {}\tq: {}\ta: {}".format(np.asarray(p).shape, np.asarray(q).shape, np.asarray(a).shape))
        inp = input("==========\n")
        inputs = np.concatenate((p, np.zeros((Config.batch_size, 1), dtype=int), q), axis=1)
        length = [len(it) for it in inputs]
        max_len = max(length)
        inputs = [padding(seq, max_len) for seq in inputs]
        return inputs, length, a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="数据集的 question 目录")
    parser.add_argument("-s", "--set", help="数据集名称")
    args = parser.parse_args()
    dt = DataHelperTest(args.root, args.set)
    dt.gen_file()

