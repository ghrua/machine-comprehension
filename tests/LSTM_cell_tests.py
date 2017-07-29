import numpy as np
import tensorflow as tf

batch = 2
dim = 3
hidden = 4

lengths = tf.placeholder(dtype=tf.int32, shape=[batch])
inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, dim])
cell = tf.nn.rnn_cell.GRUCell(hidden)
cell_state = cell.zero_state(batch, tf.float32)
output, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, initial_state=cell_state)
inputs_ = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.int32)
lengths_ = np.asarray([3, 1], dtype=np.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_ = sess.run(output, {inputs: inputs_, lengths: lengths_})
    print(output_)
