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


class MyRNN(GenericRNNModel):
    """
    The actual impl here
    """
    def __init__(selfm, keywords, winSize=100):
        """
        Put other relevant arguments here such as dim of hidden units etc.
        """
        super(MyRNN, self).__init__(keywords, winSize)
        # TODO: store init parameters in model

    def restoreFrom(self, savedFile):
        """
        savedFile: filename from which to read parameters
        """
        # TODO: read parameters from given filename
        pass

    def saveTo(self, savedFile):
        """
        savedFile: filename to which to save the current model
        """
        # TODO: save the current parameters to the given fileName
        pass

    def trainBatch(self, Xs, ys):
        """
        train on the given batch of inputs
        Xs: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        ys: numpy array of shape (batch,) denoting wordIDs of final output
        """
        # TODO: train on given batch
        pass

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
        # TODO: test on given batch
        raise NotImplementedError, "Implement me!"
