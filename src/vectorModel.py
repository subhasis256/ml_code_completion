import numpy as np
from windowModel import WindowModel
from utils import softmaxLossAndGrads, relu, drelu

class PositionDependentVectorModel(WindowModel):
    def __init__(self, keywords, winSize=100, stepsize=0.05, reg=0.01, batchsize=32,
                 wdim=32):
        super(PositionDependentVectorModel, self).__init__(keywords, winSize,
                                                           stepsize, reg, batchsize)
        # dimension of each wordvector
        self.wdim = wdim
        # initialize random parameters for weights
        totalToks = len(self.keywordList) + self.winSize
        self.add_param('W', (self.wdim*self.winSize, totalToks+1))
        self.add_param('wvec', (totalToks, self.wdim))

    def lossAndGrads(self, batchAndTargets, computeGrads=True):
        nb = len(batchAndTargets)
        ctxtoks = [ctxtok for ctxtok,tgt in batchAndTargets]
        tgts = [tgt for ctxtok,tgt in batchAndTargets]
        context = self.wvec[ctxtoks,:].reshape((nb,-1))
        scores = np.dot(context, self.W)
        preds = np.argmax(scores, axis=1)

        loss, probs, dscores = softmaxLossAndGrads(scores, tgts)
        if not computeGrads:
            return preds, probs, loss

        dW = np.dot(context.T, dscores)
        dcontext = np.dot(dscores, self.W.T).reshape((nb,-1,self.wdim))
        dwvec = np.zeros_like(self.wvec)
        for b,ctxtok in enumerate(ctxtoks):
            for w,tok in enumerate(ctxtok):
                dwvec[tok] += dcontext[b,w]
        grads = {}
        grads['W'] = dW + self.reg * self.W
        grads['wvec'] = dwvec + self.reg * self.wvec * (dwvec != 0)

        return preds, probs, loss, grads

class NonLinearVectorModel(WindowModel):
    def __init__(self, keywords, winSize=100, stepsize=0.05, reg=0.01, batchsize=32,
                 wdim=32, zdim=512):
        super(NonLinearVectorModel, self).__init__(keywords, winSize,
                                                   stepsize, reg, batchsize)
        # dimension of each wordvector
        self.wdim = wdim
        self.zdim = zdim
        # initialize random parameters for weights
        totalToks = len(self.keywordList) + self.winSize
        self.add_param('Wz', (self.wdim*self.winSize, self.zdim))
        self.add_param('Wa', (self.zdim, totalToks+1))
        self.add_param('wvec', (totalToks, self.wdim))

    def lossAndGrads(self, batchAndTargets, computeGrads=True):
        nb = len(batchAndTargets)
        ctxtoks = [ctxtok for ctxtok,tgt in batchAndTargets]
        tgts = [tgt for ctxtok,tgt in batchAndTargets]
        context = self.wvec[ctxtoks,:].reshape((nb,-1))
        z = np.dot(context, self.Wz)
        a = relu(z)
        scores = np.dot(a, self.Wa)
        preds = np.argmax(scores, axis=1)

        loss, probs, dscores = softmaxLossAndGrads(scores, tgts)
        if not computeGrads:
            return preds, probs, loss

        dWa = np.dot(a.T, dscores)
        da = np.dot(dscores, self.Wa.T)

        dz = drelu(da, a)

        dWz = np.dot(context.T, dz)
        dcontext = np.dot(dz, self.Wz.T).reshape((nb,-1,self.wdim))
        dwvec = np.zeros_like(self.wvec)
        for b,ctxtok in enumerate(ctxtoks):
            for w,tok in enumerate(ctxtok):
                dwvec[tok] += dcontext[b,w]
        grads = {}
        grads['Wa'] = dWa + self.reg * self.Wa
        grads['Wz'] = dWz + self.reg * self.Wz
        grads['wvec'] = dwvec + self.reg * self.wvec * (dwvec != 0)

        return preds, probs, loss, grads


class ConstantAttentionVectorModel(WindowModel):
    def __init__(self, keywords, winSize=100, stepsize=0.05, reg=0.01, batchsize=32,
                 wdim=32):
        super(ConstantAttentionVectorModel, self).__init__(keywords, winSize,
                                                           stepsize, reg, batchsize)
        # dimension of each wordvector
        self.wdim = wdim
        # initialize random parameters for weights
        totalToks = len(self.keywordList) + self.winSize
        self.add_param('attn', (self.winSize,), 1.)
        self.add_param('W', (self.wdim, totalToks+1))
        self.add_param('wvec', (totalToks, self.wdim))

    def lossAndGrads(self, batchAndTargets, computeGrads=True):
        nb = len(batchAndTargets)
        ctxtoks = [ctxtok for ctxtok,tgt in batchAndTargets]
        tgts = [tgt for ctxtok,tgt in batchAndTargets]

        context = np.transpose(self.wvec[ctxtoks,:], (0,2,1))
        attnvec = context.dot(self.attn)
        scores = np.dot(attnvec, self.W)
        preds = np.argmax(scores, axis=1)

        loss, probs, dscores = softmaxLossAndGrads(scores, tgts)
        if not computeGrads:
            return preds, probs, loss

        dW = np.dot(attnvec.T, dscores)
        dattnvec = np.dot(dscores, self.W.T)

        dattn = np.tensordot(context, dattnvec, axes=([0,1],[0,1]))
        dcontext = dattnvec[...,None]*self.attn

        dwvec = np.zeros_like(self.wvec)
        for b,ctxtok in enumerate(ctxtoks):
            for w,tok in enumerate(ctxtok):
                dwvec[tok] += dcontext[b,:,w]
        grads = {}
        grads['W'] = dW + self.reg * self.W
        grads['attn'] = dattn + self.reg * self.attn
        grads['wvec'] = dwvec + self.reg * self.wvec * (dwvec != 0)

        return preds, probs, loss, grads

