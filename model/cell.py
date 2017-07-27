import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMStateTuple


def linear(args, output_size, scope=None):
    all_col = [it.get_shape()[1] for it in args]
    col = 0
    for it in all_col:
        col += it.value
    args = tf.concat(args, axis=1)
    with tf.variable_scope(scope):
        weight = tf.get_variable('W', shape=[col, output_size], initializer=xavier_initializer())
        bias = tf.get_variable('b', shape=[output_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    return tf.matmul(args, weight) + bias


class LSTMCell(tf.contrib.rnn.RNNCell):
    """state is tuple"""

    def __init__(self, state_size, forget_bias=1.0):
        self._state_size = state_size
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return LSTMStateTuple(self._state_size, self._state_size)

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        c, h = state
        scope = scope or type(self).__name__
        w4m = linear([inputs, h, c], 4 * self._state_size, scope)
        print("hello")
        f, i, j, o = tf.split(w4m, self._state_size, axis=1)
        new_c = c * tf.sigmoid(f + self._forget_bias) + tf.tanh(j) * tf.sigmoid(i)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return new_h, LSTMStateTuple(new_c, new_h)


class MultiLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, cell):
        self._cell = cell

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cell)

    @property
    def output_size(self):
        return self._cell[0].output_size

    def __call__(self, inputs, states):
        cur_h = inputs
        new_states = []
        for index, cell in enumerate(self._cell):
            with tf.variable_scope("cell_%d" % index):
                cur_state = states[index]
                if index == 0:
                    cur_h = tf.zeros_like(cur_h)
                cur_h, new_sta = cell(tf.concat([inputs, cur_h], axis=1), cur_state)
                new_states.append(new_sta)
        return cur_h, tuple(new_states)
