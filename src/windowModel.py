#!/usr/bin/env python

from model import Model
import random
import numpy as np
import updates
import cPickle as pkl
from utils import colorPrint
import sys
import collections
import time
from rnnModel import GenericRNNModel

class WindowModel(GenericRNNModel):
    """
    this is a specific class of model which operates on fixed length windows of
    the source texts and tries to get the next word based on the words in the
    current window. this only contains boilerplate code useful for all such
    models.
    """
    def __init__(self, keywords, winSize=100, stepsize=0.05, reg=0.01):
        """
        @keywords: same as Model
        @winSize: length of window to consider for each word
        """
        super(WindowModel, self).__init__(keywords, winSize)
        self.params = {}
        self.stepsize = stepsize
        self.reg = reg
        self.opt = updates.AdagradOptimizer(self.stepsize)

    def add_param(self, pname, pdim, scale=0.01):
        param = scale * np.random.randn(*pdim).astype(np.float32)
        self.params[pname] = param
        setattr(self, pname, param)

    def saveTo(self, savedFile):
        pkl.dump(self, open(savedFile, 'w'))

    def restoreFrom(self, savedFile):
        saved_model = pkl.load(open(savedFile))
        for k in self.params:
            np.copyto(self.params[k], saved_model.params[k])

    def computeAccuracy(self, targets, preds):
        return np.mean(targets == preds)

    def trainBatch(self, Xs, ys):
        batch = [(X,y) for X,y in zip(Xs,ys)]
        preds, probs, loss, grads = self.lossAndGrads(batch)
        for k in grads:
            self.opt.update(self.params[k], grads[k])
        acc = self.computeAccuracy(ys, preds)
        return loss, acc

    def predictRanked(self, Xs, ys):
        batch = [(X,y) for X,y in zip(Xs,ys)]
        preds, probs, loss = self.lossAndGrads(batch, False)
        return np.argsort(-probs, axis=1)

    def getWord(self, idx, window, XWinID):
        if idx < len(self.keywordList):
            return self.IDToWord[idx]
        else:
            word = "<UNKUNK>"
            # search among the list of XWinIDs
            for ix,x in enumerate(XWinID):
                if x == idx:
                    word = window[ix]
                    break
            return word

    def predict(self, tokensTillNow):
        if len(tokensTillNow) < self.winSize:
            return ['']

        window = tokensTillNow[-self.winSize:]
        XID = self.convertToTokenIDs(window)
        XWinID, junk = self.makeWindow(XID, 0, True)

        preds, probs, loss = self.lossAndGrads([(XWinID, junk)], False)
        sortedIDs = np.argsort(-probs[0])

        sortedWords = [self.getWord(i, window, XWinID) for i in sortedIDs]

        return sortedWords

if __name__ == '__main__':
    model = WindowModel(['for', 'int', '=', '<', '>', ';', '(', ')', '{', '}'])
    print model.makeWindow([0,3,8,9,4,200,267,0,34,200,34,5,9], isoPosition=True)
