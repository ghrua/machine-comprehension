import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer

from .cell import LSTMCell, MultiLSTMCell


class DeepLSTM:
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.pred, self.loss, self.train_op = self.build_graph()

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, [None, None])
        self.labels_placeholder = tf.placeholder(tf.int32, [None])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.state_keep_prob = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, sequence_length, state_keep_prob=1.0, labels_batch=None):
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.state_keep_prob: state_keep_prob,
            self.sequence_length: sequence_length
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        embed = tf.get_variable(
            "word_embedding", shape=[self.config.vocab_size, self.config.embed_size], initializer=xavier_initializer()
        )
        embeddings = tf.nn.embedding_lookup(embed, self.input_placeholder)
        return embeddings

    def generate_cell(self):
        return rnn.DropoutWrapper(rnn.BasicLSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)

    #        return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)
    def user_defined_cell(self):
        return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)

    def build_graph(self):
        x = self.add_embedding()
        cells = [self.generate_cell() for _ in range(self.config.depth)]
        stack_cells = MultiLSTMCell(cells)
        _, state = tf.nn.dynamic_rnn(stack_cells, x, sequence_length=self.sequence_length, dtype=tf.float32)
        h = tf.concat([it[0] for it in state], axis=1, name="h")
        w_0 = tf.get_variable("w_0", [self.config.vocab_size, self.config.state_size * self.config.depth],
                              initializer=xavier_initializer())
        """
        stack_cells = self.generate_cell()  # MultiLSTMCell(cells)
        _, state = tf.nn.dynamic_rnn(stack_cells, x, sequence_length=self.sequence_length, dtype=tf.float32)
        self.h, _ = state
        w_0 = tf.get_variable("w_0", [self.config.vocab_size, self.config.state_size],
                              initializer=xavier_initializer())
        """
        b_0 = tf.get_variable("b_0", [self.config.vocab_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        pred = tf.matmul(h, w_0, transpose_b=True) + b_0

        """
        nce_loss = tf.nn.nce_loss(
            weights=w_0,
            biases=b_0,
            labels=tf.expand_dims(self.labels_placeholder, 1),
            inputs=h,
            num_sampled=self.config.num_sampled,
            num_classes=self.config.vocab_size,
            partition_strategy="div",
            name="nce_loss"
        )
        """

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred,
            labels=self.labels_placeholder,
            name="loss"
        )
        train_op = tf.train.AdadeltaOptimizer(self.config.lr).minimize(loss)
        return pred, loss, train_op

    def predict_on_batch(self, sess, inputs_batch, sequence_length):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, sequence_length=sequence_length)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def check(self):
        pass

    def train_on_batch(self, sess, inputs_batch, sequence_length, labels_batch, keep_prob=0.1):
        feed = self.create_feed_dict(
            inputs_batch=inputs_batch, sequence_length=sequence_length,
            state_keep_prob=keep_prob, labels_batch=labels_batch
        )
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
