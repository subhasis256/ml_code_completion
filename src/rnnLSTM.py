#!/usr/bin/env python

from rnnModel import GenericRNNModel
import random
import numpy as np
import updates
import cPickle as pkl
from utils import colorPrint
import sys
import collections
import time

# keras imports
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.models import model_from_json

class RnnLSTM(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(selfm, keywords, winSize=100, wdim=32,
                 lstm_activation='tanh', lstm_inner_activation='hard_sigmoid',
                 output_activation='softmax',
                 loss_optimizer='adagrad',
                 load_from_file=True, filename=""):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(RnnLSTM, self).__init__(keywords, winSize)

        if load_from_file:
            self.restoreFrom(filename)
        else:
            # save parameters
            vocab_size = keywords + winSize + 1
            self.params = {}
            self.params.wdim = wdim
            self.params.vocab_size = vocab_size

            # initialize keras model
            self.model = Sequential()
            # convert words to dense vectors 
            self.model.add(Embedding(vocab_size, wdim, input_length=winSize))
            # LSTM
            self.model.add(LSTM(wdim,
                                activation=lstm_activation,
                                inner_activation=lstm_inner_activation,
                                input_length=winSize))
            # determine output of RNN model
            self.model.add(Dense(self.vocab_size, activation=output_activation))
            # compile with optimizer, loss function
            self.model.compile(optimizer=loss_optimizer,
                               loss='categorical_crossentropy', class_mode='categorical')

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", w) as fp:
            fp.write(self.model.to_json())

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        return self.model.train_on_batch(Xs, ys, accuracy=True)

    def predictRanked(self, Xs, ys):
        """
        output a ranked prediction array for each batch, i.e.,
        output[b,r] should be the ID of the r'th top prediction for the b'th
        testing instance.
        In practice, once you get the scores for each prediction, you can simply
        do
        preds = np.argsort(-scores, axis=1)
        to get the ranked prediction array.

        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output...
        you should ideally not need this input but it is only here since I wrote
        some bad code in windowModel and I need it in the interface there. Feel
        free to ignore this parameter.
        """
        probs = self.model.predict(Xs)
        return np.argsort(-probs, axis=1)
