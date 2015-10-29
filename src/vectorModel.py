import numpy as np
from windowModel import WindowModel
from utils import softmaxLossAndGrads

class PositionDependentVectorModel(WindowModel):
    def __init__(self, keywords, winSize=100, wdim=32):
        super(PositionDependentVectorModel, self).__init__(keywords, winSize)
        # dimension of each wordvector
        self.wdim = wdim
        # initialize random parameters for weights
        totalToks = len(self.keywordList) + self.winSize
        self.add_param('W', (self.wdim*self.winSize, totalToks+1))
        self.add_param('wvec', (totalToks, self.wdim))

    def lossAndGrads(self, batchAndTargets):
        nb = len(batchAndTargets)
        ctxtoks = [ctxtok for ctxtok,tgt in batchAndTargets]
        tgts = [tgt for ctxtok,tgt in batchAndTargets]
        context = self.wvec[ctxtoks,:].reshape((nb,-1))
        scores = np.dot(context, self.W)
        preds = np.argmax(scores, axis=1)

        loss, probs, dscores = softmaxLossAndGrads(scores, tgts)
        dW = np.dot(context.T, dscores)
        dcontext = np.dot(dscores, self.W.T).reshape((nb,-1,self.wdim))
        dwvec = np.zeros_like(self.wvec)
        for b,ctxtok in enumerate(ctxtoks):
            for w,tok in enumerate(ctxtok):
                dwvec[tok] += dcontext[b,w]
        grads = {}
        grads['W'] = dW
        grads['wvec'] = dwvec

        return preds, probs, loss, grads


