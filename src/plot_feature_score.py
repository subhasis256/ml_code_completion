#!/usr/bin/env python
import cPickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import utils
from windowModel import WindowModel

label_size = 10
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
    
"""
Plot each token's score in predicting/ contributing to next token.
"""
def plotFeatureScore(score, fname):
    winSize = len(score)
    fig, ax = plt.subplots()
    plt.plot(range(winSize, 0, -1), score)
    plt.xlabel("Position of input token from the end (target)")
    plt.ylabel("Abs. correlation with target token")
    plt.savefig(fname)

class WindowTokenFeatureStrength(WindowModel):
    """
    This class ranks all tokens in a window by how much they contribute to
    predicting the next token.
    """

    def __init__(self, keywords, winSize=100, scale=0.01):
        super(WindowTokenFeatureStrength, self).__init__(keywords, winSize=winSize)
        self.scale=0.02

    """
    Compute correlation based score between next token and each token in the
    window.
    """
    def computeCorrelation(self, filesAndWords, tmpdir):
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        exy = np.zeros(self.winSize)
        ex  = np.zeros(self.winSize)
        ey  = 0
        filesAndTokens = [(name, len(tokens) - self.winSize, self.convertToTokenIDs(tokens))
                         for name, tokens in filesAndWords
                         if len(tokens) >= self.winSize + 1]
        nFiles = len(filesAndTokens)
        numWindowsTotal = sum([nws for _,nws,_ in filesAndTokens])
        for fid in range(nFiles):
            sumxy = np.zeros(self.winSize)
            sumx  = np.zeros(self.winSize)
            sumy  = 0
            numWindows = filesAndTokens[fid][1]
            tokens = filesAndTokens[fid][2]

            for w in range(numWindows):
                x_window = tokens[w:w+self.winSize]
                y_window = tokens[w+self.winSize]
                window_target = self.makeWindow(x_window, y_window, True)
                window = np.array(window_target[0]) * self.scale
                target = window_target[1] * self.scale
                sumxy = sumxy + window * target
                sumx  = sumx  + window
                sumy  = sumy  + target
            if fid % 500 == 0:
                print 'Finished computing %d files' % fid
            exy = exy + sumxy/float(numWindowsTotal)
            ex  = ex + sumx/float(numWindowsTotal)
            ey  = ey + sumy/float(numWindowsTotal)
            score = abs(exy - ex * ey)
            if fid % 2000 == 0:
                plotFeatureScore(score, '%s/correlation_%d' % (tmpdir, fid))
        return score

if __name__ == '__main__':
    # keywordfile = '../key_words/c'
    # keywords = []
    # with open(keywordfile) as fp:
    #     for line in fp:
    #         kw = re.sub(' [0-9]*$', '', line.strip())
    #         keywords.append(kw)
    # # NOTE: edit the following line to change the files input to compute
    # # feature score
    # trainfiles = utils.matchingFiles(['../data/linux'], ['c', 'h'])[0:20000]
    # trainFilesAndWords = []
    # for i,name in enumerate(trainfiles):
    #     if i % 1000 == 0:
    #         print 'Finished reading %d files' % i
    #     trainFilesAndWords.append((name,utils.tokenize(name)))
    # featureStrength = WindowTokenFeatureStrength(keywords)
    # corr_train = featureStrength.computeCorrelation(trainFilesAndWords, 'example_results/tmp_train')
    # testfiles = utils.matchingFiles(['../data/linux'], ['c', 'h'])[20000:]
    # print("Finished computing correlation on train data")
    # testFilesAndWords = []
    # for i,name in enumerate(testfiles):
    #     if i % 1000 == 0:
    #         print 'Finished reading %d files' % i
    #     testFilesAndWords.append((name,utils.tokenize(name)))
    # featureStrength = WindowTokenFeatureStrength(keywords)
    # corr_test = featureStrength.computeCorrelation(testFilesAndWords, 'example_results/tmp_test')
    # print("Finished computing correlation on test data")

    # pkl.dump(corr_train, open('example_results/tmp_train/correlation', 'w'))
    # pkl.dump(corr_test, open('example_results/tmp_test/correlation', 'w'))

    corr_train = pkl.load(open('example_results/tmp_train/correlation'))
    corr_test  = pkl.load(open('example_results/tmp_test/correlation'))

    winSize = len(corr_train)
    fig, ax = plt.subplots()
    train, = plt.plot(range(winSize, 0, -1), corr_train, color='k', label='train')
    test, = plt.plot(range(winSize, 0, -1), corr_test, color='#888888',
                     linestyle='--', label='test')
    plt.legend(['train', 'test'], loc='upper right')
    plt.xlabel("Position of input token from the end (target)", fontsize=12)
    plt.ylabel("Abs. correlation with target token", fontsize=12)
    plt.savefig("example_results/correlation.pdf")
