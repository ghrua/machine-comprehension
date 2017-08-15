import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.layers import xavier_initializer


class ImpatientReader:
    def __init__(self, config):
        self.config = config
        self.add_placeholder()
        self.embedding = self.add_embedding()

    def add_placeholders(self):
        self.question_input_placeholder = tf.placeholder(tf.int32, [None, None])
        self.context_input_placeholder = tf.placeholder(tf.int32, [None, None])
        self.question_sequence_length = tf.placeholder(tf.int32, [None])
        self.context_sequence_length = tf.placeholder(tf.int32, [None])
        self.state_keep_prob = tf.placeholder(tf.float32, shape=())
        self.labels_placeholder = tf.placeholder(tf.int32, [None])

    def add_embedding(self):
        embedding = tf.get_variable(
            "word_embedding",
            shape=[self.config.vocab_size, self.config.embed_size],
            initializer=xavier_initializer()
        )
        return embedding

    def gen_cell(self):
        cell = rnn.DropoutWrapper(
            cell=rnn.LSTMCell(self.config.state_size, True),
            state_keep_prob=self.state_keep_prob
        )
        return cell

    def context_encoder(self):
        x = tf.nn.embedding_lookup(self.embedding, self.context_input_placeholder)

        fw_cell, bw_cell = self.gen_cell(), self.gen_cell()
        output, state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, x, sequence_length=self.context_sequence_length, dtype=tf.float32
        )
        return tf.concat(output, axis=2), tf.concat([it[0] for it in state], axis=1)

    def question_decoder(self, memory, encoder_state):
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=self.config.state_size,
            memory=memory,
            memory_sequence_length=self.context_sequence_length
        )
        decoder_cell = seq2seq.AttentionWrapper(
            cell=rnn.MultiRNNCell([self.gen_cell() for _ in self.config.depth]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.config.state_size,
        )
        x = tf.nn.embedding_lookup(self.embedding, self.question_input_placeholder)
        helper = seq2seq.TrainingHelper(
            inputs=x, sequence_length=self.question_sequence_length
        )
        decoder = seq2seq.BasicDecoder(
            cell=decoder_cell, helper=helper, initial_state=encoder_state
        )
        _, final_state, _ = seq2seq.dynamic_decode(decoder=decoder)
        return final_state

    def build_graph(self):
        memory, encoder_state = self.context_encoder()
        attention_state = self.question_decoder(memory, encoder_state)

