import tensorflow as tf
from tensorflow.contrib.keras import preprocessing


def padding(sequence, maxlen, dtype='int32', padding='post', trunc='post', value=0):
    return preprocessing.sequence.pad_sequences(sequence, maxlen, dtype, padding, trunc, value)


