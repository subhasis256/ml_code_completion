#!/usr/bin/env python

from model import Model
import random
import numpy as np
import updates
import cPickle as pkl
from utils import colorPrint, WeightedRandomSampler
import sys
import collections
import time

class GenericRNNModel(Model):
    """
    Base class for RNN Models. Contains infrastructure for getting windows of
    different lengths. Everything else (creating the RNN, training, testing
    etc.) is the job of the concrete implementation
    """
    def __init__(self, keywords, winSize=100):
        """
        @keywords: same as Model
        @winSize: length of window to consider for each word
        """
        super(GenericRNNModel, self).__init__(keywords)
        self.winSize = winSize
        self.wordToID = {}
        self.IDToWord = {}
        # initialize wordToID with keyword IDs
        for i,k in enumerate(self.keywordList):
            self.wordToID[k] = i
            self.IDToWord[i] = k

    def restoreFrom(self, savedFile):
        raise NotImplementedError, "Implement me!"

    def saveTo(self, savedFile):
        raise NotImplementedError, "Implement me!"

    def trainBatch(self, Xs, ys):
        """
        X: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        y: numpy array of shape (batch,) denoting wordIDs of final output
        """
        raise NotImplementedError, "Implement me!"

    def predictRanked(self, Xs, ys):
        """
        X: numpy array of shape (batch, winSize) denoting wordIDs of inputs
        y: numpy array of shape (batch,) denoting wordIDs of final output
        """
        raise NotImplementedError, "Implement me!"

    def convertToTokenIDs(self, tokens):
        """
        convert the word tokens in @tokens to integer tokenIDs, which would be
        unique across the entire set of tokens processed. the tokenIDs are
        defined as follows:
        - tokens in keywords are given IDs from 0 to K-1, where K is #keywords
        - all other tokens are given IDs from K onwards
        """
        ids = []
        for token in tokens:
            if token not in self.wordToID:
                self.IDToWord[len(self.wordToID)] = token
                self.wordToID[token] = len(self.wordToID)
            ids.append(self.wordToID[token])
        return ids

    def makeWindow(self, tokenIDWindow, target, isoPosition=False):
        """
        converts the integer tokenIDs in @tokenIDWindow to be of the following
        form:
        - token ids in [0,K-1] are left unchanged
        - token ids >= K are converted to a contiguous range with lower IDs to
        initial elements. The way IDs are assigned here depends on the parameter
        @isoPosition. If @isoPosition=False, numbering of non-keyword tokens
        start at K. Otherwise, a non-keyword token first occuring at position w
        in the window is assigned ID K+w.
        So, for example, if K=3, and the window is [0,2,1,5,7,10,2], the
        transformed window will be [0,2,1,3,4,5,2] if @isoPosition=False, else
        it will be [0,2,1,6,7,8,2].
        converts @target using the same dictionary
        """
        nonKeywordMap = {}
        convertedTokens = [t for t in tokenIDWindow]
        for i,token in enumerate(convertedTokens):
            if token >= len(self.keywordList):
                if token not in nonKeywordMap:
                    if isoPosition:
                        windowTokenID = len(self.keywordList) + i
                    else:
                        windowTokenID = (len(self.keywordList) +
                                         len(nonKeywordMap))
                    nonKeywordMap[token] = windowTokenID
                convertedTokens[i] = nonKeywordMap[token]
        if target < len(self.keywordList):
            convertedTarget = target
        else:
            if target not in nonKeywordMap:
                convertedTarget = len(self.keywordList) + self.winSize
            else:
                convertedTarget = nonKeywordMap[target]
        return convertedTokens, convertedTarget


    def generateBatch(self, filesAndTokenIDs, randomize=True, batchsize=32):
        """
        with successive calls, generates random batches of given batchsize
        use like:
        for X,y in generateBatch(filesAndTokenIDs, 32):
            doSomething()
        """
        # filter out files with < winSize+1 tokens
        filtTokenIDs = [tokID for _,tokID in filesAndTokenIDs
                        if len(tokID) >= self.winSize+1]
        # generate a weighting array for weighting file IDs
        weights = [len(toks)-self.winSize for toks in filtTokenIDs]
        sampler = WeightedRandomSampler(weights)
        numWins = sum(weights)

#        # generate a list of all possible window beginning indices
#        numIdxs = sum([len(toks)-self.winSize for toks in filtTokenIDs])
#        beginIdxs = [(None,None) for _ in range(numIdxs)]
##        listOfBeginIdxs = [[(fid,tid) for tid in range(len(toks)-self.winSize)]
##                           for fid,toks in enumerate(filtTokenIDs)]
#        i = 0
#        for fid,toks in enumerate(filtTokenIDs):
#            for tid in range(len(toks)-self.winSize):
#                beginIdxs[i] = (fid,tid)
#                i += 1
#
##        beginIdxs = reduce(lambda x,y:x+y, listOfBeginIdxs, [])
#        if randomize:
#            random.shuffle(beginIdxs)

        for i in range(0,numWins,batchsize):
            idxs = [sampler.generate() for _ in range(batchsize)]
            widxs = [np.random.randint(len(filtTokenIDs[idx])-self.winSize)
                     for idx in idxs]
#            widxs = [0 for idx in idxs]
            Xs = [filtTokenIDs[idx][widx:widx+self.winSize]
                  for widx,idx in zip(widxs,idxs)]
            ys = [filtTokenIDs[idx][widx+self.winSize]
                  for widx,idx in zip(widxs,idxs)]
            Xys = [self.makeWindow(x,y,True) for x,y in zip(Xs,ys)]
            Xys = [(x,y) for x,y in Xys
                   if y != len(self.keywordList) + self.winSize]
            Xs = [x for x,y in Xys]
            ys = [y for x,y in Xys]
            yield np.array(Xs), np.array(ys)


    def train(self, filesAndTokens, ckpt_prefix='', batchsize=32):
        """
        @filesAndTokens: as defined in Model
        """
        # in here, we assume that we are always minimizing some loss function
        # using an sgd algorithm (may be more advanced sgd algorithms such as
        # sgd with momentum, Adam etc.) but not second order in general

        # first let us create tokenIDs corresponding to the tokens
        filesAndTokenIDs = [(name,self.convertToTokenIDs(tokens))
                            for name,tokens in filesAndTokens]
        # TODO: do minibatch based SGD using loss and gradient function defined
        # by class
        nepochs = 5

        smooth_loss = None
        smooth_known_acc = None
        smooth_abs_acc = None

        startTime = time.time()

        def smooth_update(sx, x):
            if sx is None:
                return x
            else:
                return 0.99*sx + 0.01*x

        for epoch in range(nepochs):
            b = 0
            for Xs,ys in self.generateBatch(filesAndTokenIDs, batchsize):

                # training update
                loss, acc = self.trainBatch(Xs, ys)

                smooth_loss = smooth_update(smooth_loss, loss)
                smooth_known_acc = smooth_update(smooth_known_acc, acc)
                smooth_abs_acc = smooth_update(smooth_abs_acc,
                                               acc*len(ys)/float(batchsize))

                if b % 10 == 0:
                    currentTime = time.time()
                    print '[%.3fs] Epoch %d batch %d smooth_loss %f smooth_acc %.2f%% smooth_abs_acc %.2f%%' % (currentTime-startTime, 
                                                                                                                   epoch, b,
                                                                                                                   smooth_loss,
                                                                                                                   smooth_known_acc*100,
                                                                                                                   smooth_abs_acc*100)
                if b % 100000 == 0:
                    self.saveTo('%s-%d-%d.p' % (ckpt_prefix, epoch, b))
                b += 1

    def test(self, filesAndTokens, ckpt_prefix='', batchsize=32):
        """
        @filesAndTokens: as defined in Model
        """
        # in here, we assume that we are always minimizing some loss function
        # using an sgd algorithm (may be more advanced sgd algorithms such as
        # sgd with momentum, Adam etc.) but not second order in general

        # first let us create tokenIDs corresponding to the tokens
        filesAndTokenIDs = [(name,self.convertToTokenIDs(tokens))
                            for name,tokens in filesAndTokens]
        # TODO: do minibatch based SGD using loss and gradient function defined
        # by class
        nepochs = 5

        smooth_loss = None
        smooth_known_acc = None
        smooth_abs_acc = None

        startTime = time.time()

        b = 0
        total = 0
        correct = 0
        kwcorrect = 0
        totalkw = 0
        nonkwcorrect = 0
        totalnonkw = 0
        numnonkw = 0
        nonkwcorrectrand = 0
        for Xs,ys in self.generateBatch(filesAndTokenIDs, batchsize):

            # prediction
            rankedPreds = self.predictRanked(Xs, ys)
            preds = rankedPreds[:,0]

            correct += np.sum(ys == preds)
            total += ys.shape[0]

            kwids = ys < self.winSize
            kwpreds = preds[kwids]
            kwys = ys[kwids]
            kwcorrect += np.sum(kwys == kwpreds)
            totalkw += np.sum(kwids)

            nonkwids = np.logical_and(ys >= self.winSize,
                                      ys < self.winSize + len(self.keywordList))
            nonkwys = ys[nonkwids]
            nonkwpreds = preds[nonkwids]
            nonkwcorrect += np.sum(nonkwys == nonkwpreds)
            totalnonkw += np.sum(nonkwids)

            nonkwX = Xs[nonkwids,:]
            numnonkw += np.sum(np.logical_and(nonkwX >= self.winSize,
                                              nonkwX < self.winSize + len(self.keywordList)))
            nonkwcorrectrand += np.sum(nonkwX == nonkwys[:,None])

            if b % 10 == 0:
                print 'Batch %d acc %.2f%% abs_acc %.2f%% kw_frac %.2f%% kw_acc %.2f%% non_kw_acc %.2f%% rand_non_kw_acc %.2f%%' % (b,
                                                                                                                           correct/float(total)*100,
                                                                                                                           correct/float(b*batchsize)*100,
                                                                                                                           totalkw/float(total)*100,
                                                                                                                           kwcorrect/float(totalkw)*100,
                                                                                                                           nonkwcorrect/float(totalnonkw)*100,
                                                                                                                           nonkwcorrectrand/float(numnonkw)*100)


            b += 1

