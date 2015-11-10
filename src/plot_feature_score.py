#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import re
import utils
from windowModel import WindowModel

class WindowTokenFeatureStrength(WindowModel):
    """
    This class ranks all tokens in a window by how much they contribute to
    predicting the next token.
    """

    def __init__(self, keywords, winSize=100, scale=0.02):
        super(WindowTokenFeatureStrength, self).__init__(keywords, winSize=winSize)
        self.scale=0.02

    """
    Compute correlation based score between next token and each token in the
    window.
    """
    def computeCorrelation(self, filesAndWords):
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
            if fid % 1000 == 0:
                print 'Finished reading %d files' % i
        exy = exy + sumxy/float(numWindows)
        ex  = ex + sumx/float(numWindows)
        ey  = ey + sumy/float(numWindows)
        return abs(exy - ex * ey)
    
"""
Plot each token's score in predicting/ contributing to next token.
"""
def plotFeatureScore(score, fname):
    winSize = len(score)
    plt.plot(range(winSize, 0, -1), score)
    plt.savefig(fname)

if __name__ == '__main__':
    keywordfile = '../key_words/c'
    keywords = []
    with open(keywordfile) as fp:
        for line in fp:
            kw = re.sub(' [0-9]*$', '', line.strip())
            keywords.append(kw)
    # NOTE: edit the following line to change the files input to compute
    # feature score
    datafiles = utils.matchingFiles(['../data/linux'], ['c', 'h'])[20000:30000]
    filesAndWords = []
    for i,name in enumerate(datafiles):
        if i % 1000 == 0:
            print 'Finished reading %d files' % i
            filesAndWords.append((name,utils.tokenize(name)))
    featureStrength = WindowTokenFeatureStrength(keywords)
    corr = featureStrength.computeCorrelation(filesAndWords)
    plotFeatureScore(corr, 'example_results/correlation')
