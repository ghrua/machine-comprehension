from tqdm import *
import time
from tensorflow.contrib import rnn

x = rnn.BasicLSTMCell()
rnn.MultiRNNCell()

for i in tqdm(range(100)):
    time.sleep(.01)
