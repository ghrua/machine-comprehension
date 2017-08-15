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
    vocab_size = 100002
    embed_size = 200
    state_size = 256
    depth = 2
    lr = 5E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 200


class Train:
    def __init__(self, root_dir, config, debug=True):
        self.config = config
        self.test_set = tf.gfile.Glob(join(root_dir, "test", "*.question"))
        self.training_set = tf.gfile.Glob(join(root_dir, "training", "*.question"))
        self.validation_set = tf.gfile.Glob(join(root_dir, "validation", "*.question"))
        self.vocabulary, self.reverse_vocabulary = load_vocab(root_dir, str(Config.vocab_size - 2))
        self.reverse_vocabulary = ['BAR_', 'UNK_'] + self.reverse_vocabulary
        self.debug = debug

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
        else:
            size = 5 * self.config.batch_size
        samples = random.sample(list(data_set), size)
        p, q, a = zip(*questions_to_token_ids(samples, self.vocabulary))
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

    def train(self, mc_model, model_output):
        if self.debug:
            data_set = self.validation_set
            random.seed(10)
        else:
            data_set = self.training_set
        with tf.Graph().as_default():
            max_f1_score = 0.0
            model = mc_model(self.config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                for epoch in range(self.config.n_epoch):
                    pbar = tqdm(range(self.config.train_steps), desc="{} epoch".format(epoch))
                    for _ in pbar:
                        inputs, length, labels = self.gen_file(data_set)
                        loss, _ = model.train_on_batch(session, inputs, length, labels)
                        pbar.set_description("loss: {:.2f}".format(np.mean(loss)))
                    inputs, length, labels = self.gen_file(self.test_set, False)
                    prediction = model.predict_on_batch(session, inputs, length)
                    f1 = f1_score(labels, prediction, average='micro')
                    print("evaluate: acc/pre/rec/f1: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                        accuracy_score(labels, prediction),
                        precision_score(labels, prediction, average='micro'),
                        recall_score(labels, prediction, average='micro'),
                        f1
                    ))
                    if f1 > max_f1_score:
                        print("New best score! Saving model in {}".format(model_output))
                        max_f1_score = f1
                        saver.save(session, model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="数据集的 question 目录")
    args = parser.parse_args()
    config = Config()
    train = Train(args.root, config, debug=False)
    train.train(mc_model=DeepLSTM, model_output=join(args.root, "model.ckpt"))
