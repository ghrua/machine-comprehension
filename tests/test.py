from tqdm import *
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from model.cell import LSTMCell, MultiLSTMCell


def user_lstm(keep_prob):
    return rnn.DropoutWrapper(LSTMCell(4), state_keep_prob=keep_prob)


def tensor_lstm():
    return rnn.BasicLSTMCell(4)


if __name__ == "__main__":
    keep_prob = tf.placeholder(tf.float32, shape=())
    # cell = [user_lstm(keep_prob) for _ in range(2)]
    fw_cell = tensor_lstm()
    bw_cell = tensor_lstm()
    mode = tf.placeholder(tf.int32, shape=())
    inputs_placeholder = tf.placeholder(tf.float32, [None, None, 3])
    sequence_length = tf.placeholder(tf.int32, [None])
    output, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_placeholder, sequence_length, dtype=tf.float32)
    output = tf.concat(output, axis=2)
    state = tf.concat([it[0] for it in state], axis=1)
#    state = tf.concat(state, axis=1)

    with tf.Session() as sess:
        inputs = np.asarray(
            [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
             [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
            dtype=np.int32
        )
        lengths = np.asarray([3, 1], dtype=np.int32)
        keep = 1.0
        init = tf.global_variables_initializer()
        sess.run(init)
        o1 = sess.run(state, feed_dict={inputs_placeholder: inputs, sequence_length: lengths, keep_prob: keep, mode: 1})
        print(o1)
        # print(s)
