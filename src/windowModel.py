#!/usr/bin/env python

from model import Model

class WindowModel(Model):
    """
    this is a specific class of model which operates on fixed length windows of
    the source texts and tries to get the next word based on the words in the
    current window. this only contains boilerplate code useful for all such
    models.
    """
    def __init__(self, keywords, winSize=100):
        """
        @keywords: same as Model
        @winSize: length of window to consider for each word
        """
        super(WindowModel, self).__init__(keywords)
        self.winSize = winSize
        self.wordToID = {}
        # initialize wordToID with keyword IDs
        for i,k in enumerate(self.keywordList):
            self.wordToID[k] = i

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
                self.wordToID[token] = len(self.wordToID)
            ids.append(self.wordToID[token])
        return ids

    def makeWindow(self, tokenIDWindow, isoPosition=False):
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
        return convertedTokens

    def train(self, filesAndTokens):
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


if __name__ == '__main__':
    model = WindowModel(['for', 'int', '=', '<', '>', ';', '(', ')', '{', '}'])
    print model.makeWindow([0,3,8,9,4,200,267,0,34,200,34,5,9], isoPosition=True)
