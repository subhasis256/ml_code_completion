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
import theano

# keras imports
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Flatten, RepeatVector, Reshape, Permute
from keras.layers.convolutional import Convolution1D
from keras.regularizers import l2
from keras.models import Sequential, Graph
from keras.models import model_from_json

class RnnLSTM(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(self, keywords, winSize=100, wdim=32, zdim=1024, reg=0.,
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
            vocab_size = len(keywords) + winSize + 1
            self.params = {}
            self.params["wdim"] = wdim
            self.params["vocab_size"] = vocab_size

            # initialize keras model
            self.model = Sequential()
            # convert words to dense vectors 
            self.model.add(Embedding(vocab_size, wdim, input_length=winSize))
            # LSTM
            self.model.add(GRU(zdim,
#                               activation=lstm_activation,
#                               inner_activation=lstm_inner_activation,
                               input_length=winSize))
            # determine output of RNN model
            self.model.add(Dense(vocab_size, activation=output_activation,
                                 W_regularizer=l2(reg)))
            # compile with optimizer, loss function
            self.model.compile(optimizer=loss_optimizer,
                               loss='categorical_crossentropy', class_mode='categorical')

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(savedFilePrefix + "_weights", overwrite=True)

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        B = ys.shape[0]
        ys_labels = np.zeros((B, self.params["vocab_size"]),
                             dtype=np.int)
        ys_labels[np.arange(B), ys] = 1
        return self.model.train_on_batch(Xs, ys_labels, accuracy=True)

    def score(self, Xs):
        return self.model.predict(Xs)

class RnnDense(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(self, keywords, winSize=100, wdim=32, zdim=1024, reg=0.,
                 lstm_activation='tanh', lstm_inner_activation='hard_sigmoid',
                 output_activation='softmax',
                 loss_optimizer='adagrad',
                 load_from_file=True, filename=""):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(RnnDense, self).__init__(keywords, winSize)

        # save parameters
        vocab_size = len(keywords) + winSize + 1
        self.params = {}
        self.params["wdim"] = wdim
        self.params["vocab_size"] = vocab_size

        # initialize keras model
        self.model = Sequential()
        # convert words to dense vectors 
        self.model.add(Embedding(vocab_size, wdim, input_length=winSize))
        # flatten
        self.model.add(Flatten())
        # determine output of RNN model
        self.model.add(Dense(zdim, activation='relu', W_regularizer=l2(reg)))
        self.model.add(Dense(vocab_size, activation=output_activation,
                             W_regularizer=l2(reg)))
        # compile with optimizer, loss function
        self.model.compile(optimizer=loss_optimizer,
                           loss='categorical_crossentropy', class_mode='categorical')

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(savedFilePrefix + "_weights", overwrite=True)

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        B = ys.shape[0]
        ys_labels = np.zeros((B, self.params["vocab_size"]),
                             dtype=np.int)
        ys_labels[np.arange(B), ys] = 1
        return self.model.train_on_batch(Xs, ys_labels, accuracy=True)

    def score(self, Xs):
        return self.model.predict(Xs)

class RnnDense2(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(self, keywords, winSize=100, wdim=32, zdim=1024, zdim2=1024, reg=0.,
                 lstm_activation='tanh', lstm_inner_activation='hard_sigmoid',
                 output_activation='softmax',
                 loss_optimizer='adagrad',
                 load_from_file=True, filename=""):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(RnnDense2, self).__init__(keywords, winSize)

        # save parameters
        vocab_size = len(keywords) + winSize + 1
        self.params = {}
        self.params["wdim"] = wdim
        self.params["vocab_size"] = vocab_size

        # initialize keras model
        self.model = Sequential()
        # convert words to dense vectors 
        self.model.add(Embedding(vocab_size, wdim, input_length=winSize))
        # flatten
        self.model.add(Flatten())
        # determine output of RNN model
        self.model.add(Dense(zdim, activation='relu', W_regularizer=l2(reg)))
        self.model.add(Dense(zdim2, activation='relu', W_regularizer=l2(reg)))
        self.model.add(Dense(vocab_size, activation=output_activation,
                             W_regularizer=l2(reg)))
        # compile with optimizer, loss function
        self.model.compile(optimizer=loss_optimizer,
                           loss='categorical_crossentropy', class_mode='categorical')

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(savedFilePrefix + "_weights", overwrite=True)

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        B = ys.shape[0]
        ys_labels = np.zeros((B, self.params["vocab_size"]),
                             dtype=np.int)
        ys_labels[np.arange(B), ys] = 1
        return self.model.train_on_batch(Xs, ys_labels, accuracy=True)

    def score(self, Xs):
        return self.model.predict(Xs)

class RnnAttentionDense2(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(self, keywords, winSize=100, wdim=32, zdim=1024, zdim2=1024, reg=0.,
                 lstm_activation='tanh', lstm_inner_activation='hard_sigmoid',
                 output_activation='softmax',
                 loss_optimizer='adagrad',
                 load_from_file=True, filename=""):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(RnnAttentionDense2, self).__init__(keywords, winSize)

        # save parameters
        vocab_size = len(keywords) + winSize + 1
        self.params = {}
        self.params["wdim"] = wdim
        self.params["vocab_size"] = vocab_size

        # initialize keras model
        self.model = Graph()
        # convert words to dense vectors
        self.model.add_input(name='word', input_shape=(winSize,), dtype='int')
        self.model.add_node(Embedding(vocab_size, wdim, input_length=winSize),
                            name='wvec', input='word')
        self.model.add_node(Flatten(),
                            name='wvecf', input='wvec')
        self.model.add_node(Dense(winSize, activation='sigmoid',
                                  W_regularizer=l2(reg)),
                            name='attn', input='wvecf')
        self.model.add_node(RepeatVector(wdim),
                            name='attnr', input='attn')
        self.model.add_node(Permute(dims=(2,1)),
                            name='attnp', input='attnr')
        # multiply word vector by attention and flatten output
        self.model.add_node(Flatten(), name='awvecf', inputs=['wvec', 'attnp'], merge_mode='mul')
        # fully connected layers
        self.model.add_node(Dense(zdim, activation='relu',
                                  W_regularizer=l2(reg)),
                            name='d1', input='awvecf')
        self.model.add_node(Dense(zdim2, activation='relu',
                                  W_regularizer=l2(reg)),
                            name='d2', input='d1')
        # final layer
        self.model.add_node(Dense(vocab_size, activation=output_activation,
                                  W_regularizer=l2(reg)),
                            name='d3', input='d2')

        self.model.add_output(name='probs', input='d3')
        # compile with optimizer, loss function
        self.model.compile(loss_optimizer,
                           {'probs': 'categorical_crossentropy'})

        # also compile a function for getting the attention vector
        self.get_attn = theano.function([self.model.inputs[i].input for i in
                                         self.model.input_order],
                                        self.model.nodes['attn'].get_output(train=False),
                                        on_unused_input='ignore')


    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

        # also compile a function for getting the attention vector
        print [self.model.inputs[i].input for i in self.model.input_order]
        self.get_attn = theano.function([self.model.inputs[i].input for i in
                                         self.model.input_order],
                                        self.model.nodes['attn'].get_output(train=False),
                                        on_unused_input='ignore')

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(savedFilePrefix + "_weights", overwrite=True)

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        B = ys.shape[0]
        probs = self.model.predict({'word': Xs})['probs']
        acc = np.mean(np.argmax(probs, axis=1) == ys)
        ys_labels = np.zeros((B, self.params["vocab_size"]),
                             dtype=np.int)
        ys_labels[np.arange(B), ys] = 1
        loss = self.model.train_on_batch({'word': Xs,
                                          'probs': ys_labels})
        if isinstance(loss, list):
            loss = loss[0]
        return loss, acc

    def score(self, Xs):
        return self.model.predict({'word': Xs})['probs']

    def attention(self, Xs):
        attn = self.get_attn(Xs.astype(np.int32))
        return attn

class RnnConvDense(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(self, keywords, winSize=100, wdim=32, kSize=7, convdim=64, zdim=1024, reg=0.,
                 lstm_activation='tanh', lstm_inner_activation='hard_sigmoid',
                 output_activation='softmax',
                 loss_optimizer='adagrad',
                 load_from_file=True, filename=""):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(RnnConvDense, self).__init__(keywords, winSize)

        # save parameters
        vocab_size = len(keywords) + winSize + 1
        self.params = {}
        self.params["wdim"] = wdim
        self.params["vocab_size"] = vocab_size

        # initialize keras model
        self.model = Sequential()
        # convert words to dense vectors 
        self.model.add(Embedding(vocab_size, wdim, input_length=winSize))
        # determine output of RNN model
        self.model.add(Convolution1D(convdim, kSize, border_mode='same', activation='relu', W_regularizer=l2(reg)))
        # flatten
        self.model.add(Flatten())
        self.model.add(Dense(zdim, activation='relu', W_regularizer=l2(reg)))
#        self.model.add(Dense(zdim2, activation='relu', W_regularizer=l2(reg)))
        self.model.add(Dense(vocab_size, activation=output_activation,
                             W_regularizer=l2(reg)))
        # compile with optimizer, loss function
        self.model.compile(optimizer=loss_optimizer,
                           loss='categorical_crossentropy', class_mode='categorical')

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

    def saveTo(self, savedFilePrefix):
        """
        savedFile: filename to which to save the current model
        """
        # try pickle if this doesn't work 
        pkl.dump(self.params, open(savedFilePrefix + "_params", 'w'))
        with open(savedFilePrefix + "_model", 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(savedFilePrefix + "_weights", overwrite=True)

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        return value: loss (using accuracy=True)
        """
        B = ys.shape[0]
        ys_labels = np.zeros((B, self.params["vocab_size"]),
                             dtype=np.int)
        ys_labels[np.arange(B), ys] = 1
        return self.model.train_on_batch(Xs, ys_labels, accuracy=True)

    def score(self, Xs):
        return self.model.predict(Xs)

class GenericKerasModel(GenericRNNModel):
    def __init__(self, keywords, winSize=100):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(GenericKerasModel, self).__init__(keywords, winSize)

    def restoreFrom(self, savedFilePrefix):
        """
        savedFile: filename from which to read parameters
        """
        # try pickle if this doesn't work 
        self.params = pkl.load(open(savedFilePrefix + "_params"))
        with open(savedFilePrefix + "_model") as fp:
            self.model = model_from_json(fp.read())
        self.model.load_weights(savedFilePrefix + "_weights")

    def score(self, Xs):
        if isinstance(self.model, Graph):
            return self.model.predict({'word': Xs})['probs']
        else:
            return self.model.predict(Xs)

class EnsembleModel(GenericRNNModel):
    def __init__(self, keywords, winSize=100):
        """
        @keywords: same as Model
        @winSize: length of window to consider for each word
        """
        super(EnsembleModel, self).__init__(keywords, winSize)
        self.keywords = keywords
        self.winSize = winSize

    def restoreFrom(self, savedFilePrefixes):
        self.models = []
        for savedFilePrefix in savedFilePrefixes.split(":"):
            model = GenericKerasModel(self.keywords, self.winSize)
            model.restoreFrom(savedFilePrefix)
            self.models.append(model)

    def score(self, Xs):
        return np.sum([model.score(Xs) for model in self.models], axis=0)

    def attention(self, Xs):
        return np.mean([model.attention(Xs) for model in self.models], axis=0)

