#!/usr/bin/env python

from utils import colorPrint
import sys

class Model(object):
    """
    This is a base class for holding any kind of model that we create.
    It should take in a list of tuples of the following form:
    (filename, list of tokens in file)
    and a list of keywords
    and build a model starting from there.
    """
    def __init__(self, keywords):
        """
        @keywords: list of keywords for this instance
        store the list of keywords in self.keywordList
        and a corresponding frozenset in self.keywordSet
        """
        self.keywordList = keywords
        self.keywordSet = frozenset(keywords)

    def train(self, filesAndTokens):
        """
        @filesAndTokens: list of tuples of the form 
        (fileName, list of tokens in file)
        """
        raise NotImplementedError, "Implement me!"

    def test(self, filesAndTokens):
        """
        @filesAndTokens: list of tuples of the form 
        (fileName, list of tokens in file)
        """
        raise NotImplementedError, "Implement me!"

    def predict(self, tokensTillNow):
        """
        Same as above, only this time predict a ranked list of tokens
        """
        raise NotImplementedError, "Implement me!"

    def testOverlap(self, fileTokens):
        """
        Here, we will iterate through the tokens, and will get the prediction at
        each character granularity (i.e. filter predictions based on character
        level match after each character)
        Returns the character level annotations for each token (partially
        correct, correct or wrong), and also the top 5 candidates for each
        token.
        """
        annotations = []
        candidates = []
        attentions = []
        for tid in range(len(fileTokens)):
            tokensTillNow = fileTokens[:tid]
            nextTok = fileTokens[tid]
            sortedPreds, attention = self.predict(tokensTillNow)
            candidates.append(sortedPreds[:5])
            attentions.append(attention)
            tokenAnn = []
            for cid in range(len(nextTok)):
                filteredPred = [w for w in sortedPreds
                                if w[:cid] == nextTok[:cid]]
                if len(filteredPred) > 0:
                    if filteredPred[0] == nextTok:
                        tokenAnn.append(2)
                    elif nextTok in filteredPred[:5]:
                        tokenAnn.append(1)
                    else:
                        tokenAnn.append(0)
                else:
                    tokenAnn.append(0)
            annotations.append(tokenAnn)
        return annotations, candidates, attentions


if __name__ == '__main__':
    import random

    # an example implementation of Model
    # in this implementation we always predict a random keyword from the list of
    # history tokens
    class CrappyModel(Model):
        def __init__(self, keywords, history=10):
            super(CrappyModel, self).__init__(keywords)
            self.history = history

        def train(self, filesAndTokens):
            # training is easy-peasy when we have nothing to do! :)
            return

        def predict(self, fileTillNow):
            historyTokens = fileTillNow[-self.history:]
            # get all the keywords in that list
            historyKeywords = [w for w in historyTokens if w in self.keywordSet]
            if len(historyKeywords) == 0:
                return ""
            else:
                return random.choice(historyKeywords)


    # let's do some crappy tests on our crappy model
    model = CrappyModel(['for', 'int', '=', '<', '>', ';', '(', ')', '{', '}'])
    train_insts = [
        ('foo.c', ['mary' , 'had', 'a', 'little', 'lamb', 'little', 'lamb',
                   'little', 'lamb']),
        ('bar.c', ['for', '(', 'int', 'i', '=', '0', ';', 'i', '<', 'x'])]
    model.train(train_insts)
    print model.predict(['for', '(', 'int', 'i', '=', '0', ';'])
