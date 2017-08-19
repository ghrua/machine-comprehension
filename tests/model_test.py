import sys
import random
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from os.path import abspath, dirname, join, exists
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils import padding
from data_utils import questions_to_token_ids, load_vocab
from model.deep_lstm import DeepLSTM


class Config:
    vocab_size = 6
    embed_size = 200
    state_size = 256
    depth = 2
    lr = 1E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 30

TRAINING_data = [
    [[1, 2, 3, 4, 5, 1, 1, 1], 8, 1],
    [[1, 2, 3, 4, 5, 2, 2, 2], 8, 2],
    [[1, 2, 3, 4, 5, 3, 3, 3], 8, 3],
    [[5, 4, 3, 2, 1, 3, 3, 3], 8, 1],
    [[5, 4, 3, 2, 1, 2, 2, 2], 8, 2],
]
TEST_data = [
    [[1, 3, 2, 4, 5, 1, 1, 2], 8, 1],
    [[1, 4, 2, 3, 5, 1, 2, 2], 8, 2],
    [[2, 2, 3, 4, 4, 3, 3, 3], 8, 3]
]

class Train:
    def __init__(self, config):
        self.config = config
        self.test_set = TEST_data
        self.training_set = TRAINING_data

    """
    def test_labels(self):
        _, _, a = zip(*questions_to_token_ids(self.training_set, self.vocabulary))
        for i in range(len(a)):
            if a[i] > 100001:
                print(a[i], self.training_set[i])
    """

    def gen_file(self, data_set, size=True):
        if size:
            size = self.config.batch_size
            samples = random.sample(list(data_set), size)
        else:
            samples = data_set
        p, q, a = zip(samples)
        inputs = [q[i] + [0] + p[i] for i in range(size)]
        length = [len(it) for it in inputs]
        max_len = max(length)
        inputs = padding(inputs, max_len)
        for row in inputs:
            for col in row:
                if col < 0 or col >= Config.vocab_size:
                    print("BUG!!!!!! index: {}".format(col))
        labels = np.array(a).flatten()
        length = np.array(length)
        return inputs, length, labels

    def train(self, mc_model):
        with tf.Graph().as_default():
            model = mc_model(self.config)
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                for epoch in range(self.config.n_epoch):
                    pbar = tqdm(range(self.config.train_steps), desc="{} epoch".format(epoch))
                    for _ in pbar:
                        inputs, length, labels = zip(*self.training_set)
                        loss, prediction = model.train_on_batch(session, inputs, length, labels)
                        pbar.set_description("loss: {:.2f} acc: {:.2f}".format(np.mean(loss), accuracy_score(labels, prediction)))
                    inputs, length, labels = zip(*self.test_set)
                    prediction = model.predict_on_batch(session, inputs, length)
                    f1 = f1_score(labels, prediction, average='micro')
                    print("evaluate: acc/pre/rec/f1: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                        accuracy_score(labels, prediction),
                        precision_score(labels, prediction, average='micro'),
                        recall_score(labels, prediction, average='micro'),
                        f1
                    ))


if __name__ == "__main__":
    config = Config()
    train = Train(config)
    train.train(mc_model=DeepLSTM)
