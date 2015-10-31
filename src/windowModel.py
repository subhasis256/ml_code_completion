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

class WindowModel(Model):
    """
    this is a specific class of model which operates on fixed length windows of
    the source texts and tries to get the next word based on the words in the
    current window. this only contains boilerplate code useful for all such
    models.
    """
    def __init__(self, keywords, winSize=100, stepsize=0.05, batchsize=32):
        """
        @keywords: same as Model
        @winSize: length of window to consider for each word
        """
        super(WindowModel, self).__init__(keywords)
        self.winSize = winSize
        self.wordToID = {}
        self.IDToWord = {}
        # initialize wordToID with keyword IDs
        for i,k in enumerate(self.keywordList):
            self.wordToID[k] = i
            self.IDToWord[i] = k
        self.params = {}
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.opt = updates.AdagradOptimizer(self.stepsize)

    def add_param(self, pname, pdim, scale=0.01):
        param = scale * np.random.randn(*pdim).astype(np.float32)
        self.params[pname] = param
        setattr(self, pname, param)

    def restoreFrom(self, savedFile):
        saved_model = pkl.load(open(savedFile))
        for k in self.params:
            np.copyto(self.params[k], saved_model.params[k])

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

    def computeAccuracy(self, targets, preds):
        return np.mean(targets == preds)

    def train(self, filesAndTokens, ckpt_prefix=''):
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
        totalTokens = sum([len(tokens) for name,tokens in filesAndTokens])
        batches = totalTokens / self.batchsize
        Nfiles = len(filesAndTokens)
        print totalTokens, batches

        smooth_loss = None
        smooth_known_acc = None
        smooth_abs_acc = None

        startTime = time.time()

        for epoch in range(nepochs):
            for b in range(batches):
                # get a range of random self.batchsize sized beginning indexes
                fileIDs = [random.randrange(Nfiles) for _ in range(self.batchsize)]
                filtFileIDs = [fid for fid in fileIDs
                               if len(filesAndTokens[fid][1]) > self.winSize+1]
                begins = [random.randrange(len(filesAndTokens[fid][1])-self.winSize-1)
                          for fid in filtFileIDs]
                batch = [filesAndTokenIDs[fid][1][begin:begin+self.winSize]
                         for fid,begin in zip(filtFileIDs,begins)]
                targets = [filesAndTokenIDs[fid][1][begin+self.winSize]
                           for fid,begin in zip(filtFileIDs,begins)]
                batchAndTargets = [self.makeWindow(tokenIDs,target,True)
                                   for tokenIDs,target in zip(batch,targets)]
                knownBatch = [(toks,tgt) for toks,tgt in batchAndTargets
                              if tgt != len(self.keywordList) + self.winSize]
                preds, probs, loss, grads = self.lossAndGrads(knownBatch)
                for k in grads:
                    self.opt.update(self.params[k], grads[k])

                tokTgts = np.array([tgt for toks,tgt in knownBatch])
                acc = self.computeAccuracy(tokTgts, preds)

                if smooth_loss is None:
                    smooth_loss = loss
                    smooth_known_acc = acc
                    smooth_abs_acc = acc * tokTgts.shape[0]/float(self.batchsize)
                else:
                    smooth_loss = 0.99*smooth_loss + 0.01*loss
                    smooth_known_acc = 0.99*smooth_known_acc + 0.01*acc
                    smooth_abs_acc = 0.99*smooth_abs_acc + 0.01*acc*tokTgts.shape[0]/float(self.batchsize)

                if b % 10 == 0:
                    currentTime = time.time()
                    print '[%.3fs] Epoch %d batch %d/%d smooth_loss %f smooth_acc %.2f%% smooth_abs_acc %.2f%%' % (currentTime-startTime, 
                                                                                                                   epoch, b, batches,
                                                                                                                   smooth_loss,
                                                                                                                   smooth_known_acc*100,
                                                                                                                   smooth_abs_acc*100)
                if b % 100000 == 0:
                    pkl.dump(self, open('%s-%d-%d.p' % (ckpt_prefix, epoch, b), 'w'))

    def printPreds(self, XyIDs, XyWinIDs, preds, probs):
        assert len(XyWinIDs) == len(XyIDs)
        for i in range(len(XyIDs)):
            XID, yID = XyIDs[i]
            XWinID, yWinID = XyWinIDs[i]
            pred = preds[i]

            words = [self.IDToWord[x] for x in XID]
            yWord = self.IDToWord[yID]

            predWord = "<UNKUNK>"
            if pred < len(self.keywordList):
                predWord = self.IDToWord[pred]
            elif pred < len(self.keywordList) + self.winSize:
                # search among the list of XWinIDs
                for ix,x in enumerate(XWinID):
                    if x == pred:
                        predWord = self.IDToWord[XID[ix]]
                        break

            if not all([c.isalpha() or c.isdigit() or c == '_' for c in yWord]):
                continue

            colorPrint(' '.join(words))
            colorPrint(XID)
            if pred == yWinID:
                color = 'green'
                rank = 1
            elif np.any(yWinID == np.argsort(-probs[i])[:5]):
                color = 'yellow'
                rank = np.where(yWinID == np.argsort(-probs[i])[:5])[0][0]+1
            else:
                color = 'red'
                rank = 'INF'
            colorPrint(yWord, '   ', predWord, ' rank:', rank, 'p:', probs[i,yWinID], color=color)
            sys.stdout.write('\n')
            if pred == yWinID and predWord != yWord:
                print XID, yID, XWinID, yWinID, pred
                print self.keywordList[pred], self.IDToWord[pred]
                raise AssertionError, "somethings not right"

    def test(self, filesAndTokens):
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
        self.batchsize = 32
        totalTokens = sum([len(tokens) for name,tokens in filesAndTokens])
        batches = totalTokens / self.batchsize
        Nfiles = len(filesAndTokens)
        print totalTokens, batches

        total_loss = 0
        total_acc = 0
        total_tgts = 0
        abs_tgts = 0

        kw_correct = 0
        total_kw = 0

        total_non_kw = 0
        num_non_kw = 0
        non_kw_correct = 0
        non_kw_correct_rand = 0

        for b in range(batches):
            # get a range of random self.batchsize sized beginning indexes
            fileIDs = [random.randrange(Nfiles) for _ in range(self.batchsize)]
            filtFileIDs = [fid for fid in fileIDs
                           if len(filesAndTokens[fid][1]) > self.winSize+1]
            begins = [random.randrange(len(filesAndTokens[fid][1])-self.winSize-1)
                      for fid in filtFileIDs]
            XIDs = [filesAndTokenIDs[fid][1][begin:begin+self.winSize]
                    for fid,begin in zip(filtFileIDs,begins)]
            yIDs = [filesAndTokenIDs[fid][1][begin+self.winSize]
                    for fid,begin in zip(filtFileIDs,begins)]

            XyWinIDs = [self.makeWindow(XID,yID,True)
                        for XID,yID in zip(XIDs,yIDs)]
            filtXyWinIDs = [(XWinID,yWinID) for XWinID,yWinID in XyWinIDs
                            if yWinID != len(self.keywordList) + self.winSize]
            filtXyIDs = [(XID,yID)
                         for XID,yID,(XWinID,yWinID) in zip(XIDs,yIDs,XyWinIDs)
                         if yWinID != len(self.keywordList) + self.winSize]
            preds, probs, loss, grads = self.lossAndGrads(filtXyWinIDs)

            yWinIDs = np.array([yWinID for XWinID,yWinID in filtXyWinIDs])
            XWinIDs = np.array([XWinID for XWinID,yWinID in filtXyWinIDs])
            acc = self.computeAccuracy(yWinIDs, preds)

#            self.printPreds(filtXyIDs, filtXyWinIDs, preds, probs)

            kw_targets = yWinIDs < self.winSize
            kw_preds = preds[kw_targets]

            kw_correct += np.sum(kw_preds == yWinIDs[kw_targets])
            total_kw += np.sum(kw_targets)

            non_kw_targets = np.logical_and(yWinIDs >= self.winSize,
                                            yWinIDs < self.winSize + len(self.keywordList))
            non_kw_preds = preds[non_kw_targets]
            non_kw_correct += np.sum(non_kw_preds == yWinIDs[non_kw_targets])
            total_non_kw += np.sum(non_kw_targets)

            non_kw_X = XWinIDs[non_kw_targets,:]
            num_non_kw += np.sum(np.logical_and(non_kw_X >= self.winSize,
                                                non_kw_X < self.winSize + len(self.keywordList)))
            non_kw_correct_rand += np.sum(non_kw_X == non_kw_preds[:,None])

            total_tgts += preds.shape[0]
            abs_tgts += self.batchsize
            total_loss += loss * preds.shape[0]
            total_acc += acc * preds.shape[0]

            if b % 10 == 0:
                print 'Batch %d/%d loss %f acc %.2f%% abs_acc %.2f%% kw_frac %.2f%% kw_acc %.2f%% non_kw_acc %.2f%% rand_non_kw_acc %.2f%%' % (b, batches,
                                                                                                                           total_loss/total_tgts,
                                                                                                                           total_acc/total_tgts*100,
                                                                                                                           total_acc/abs_tgts*100,
                                                                                                                           total_kw/float(total_tgts)*100,
                                                                                                                           kw_correct/float(total_kw)*100,
                                                                                                                           non_kw_correct/float(total_non_kw)*100,
                                                                                                                           non_kw_correct_rand/float(num_non_kw)*100)


    def word(self, ID):
        return self.IDToWord[ID] if ID in self.IDToWord else "<UNKUNK>"

    def multiPredict(self, XID):
        nsteps = 20
        beam = 100

        paths = [self.makeWindow(XID,0,True)]
        pathProbs = np.array([1.])
        print paths[0][0]
        for step in range(nsteps):
            preds, probs, _, _ = self.lossAndGrads(paths)
            newPathProbs = probs[:,:-1] * pathProbs[:,None]
            highestProbIdxs = np.argsort(newPathProbs, axis=None)[-beam:]
            highestProbPathEnds = np.unravel_index(highestProbIdxs,
                                                   newPathProbs.shape)
            highestProbs = newPathProbs.flatten()[highestProbIdxs]
            highestProbs /= np.sum(highestProbs)
            newBestPaths = [paths[xid][0][1:] + [e]
                            for xid,e in zip(*highestProbPathEnds)]
            paths = [(p,0) for p in newBestPaths]
            pathProbs = highestProbs
            endProbs = collections.defaultdict(float)
            for prob,path in zip(pathProbs,newBestPaths):
                endProbs[path[-1]] += prob
            print max([(p,self.word(e)) for e,p in endProbs.items()])

if __name__ == '__main__':
    model = WindowModel(['for', 'int', '=', '<', '>', ';', '(', ')', '{', '}'])
    print model.makeWindow([0,3,8,9,4,200,267,0,34,200,34,5,9], isoPosition=True)
