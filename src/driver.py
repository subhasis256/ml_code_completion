import fnmatch as fn
import os
import re
from vectorModel import *
import utils
import random

if __name__ == '__main__':
    import cPickle as pkl
    import sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        choices=['train', 'test', 'predict'],
                        help='train/test/predict')
    parser.add_argument('--restore',
                        help='file from which to restore parameters for testing or resuming training')
    parser.add_argument('--ckpt',
                        default='',
                        help='prefix of checkpoint files')
    parser.add_argument('--win',
                        type=int,
                        default=40,
                        help='window size')
    parser.add_argument('--dim',
                        type=int,
                        default=32,
                        help='word vector dimensions')
    parser.add_argument('--zdim',
                        type=int,
                        default=512,
                        help='intermediate vector dimensions')
    parser.add_argument('--keywords',
                        default='../key_words/c',
                        help='file containing key words and their frequencies')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05,
                        help='learning rate')
    parser.add_argument('--reg',
                        type=float,
                        default=0.0,
                        help='regularization constant')
    parser.add_argument('--batch',
                        type=int,
                        default=32,
                        help='batch size')
    args = parser.parse_args()

    files = utils.matchingFiles(['../data/linux'], ['c', 'h'])
    filesAndTokens = []

    if args.command == 'train':
        fileSubset = files[:20000]
    elif args.command == 'test':
        fileSubset = random.sample(files[20000:], 2000)
    else:
        fileSubset = files[20000:]

    if args.command == 'train' or args.command == 'test':
        for i,name in enumerate(fileSubset):
            if i % 1000 == 0:
                print '%d files done' % i
            filesAndTokens.append((name,utils.tokenize(name)))
        print len(filesAndTokens)
        print sum([len(tokens) for name,tokens in filesAndTokens])

    keywords = []
    with open(args.keywords) as fp:
        for line in fp:
            kw = re.sub(' [0-9]*$', '', line.strip())
            keywords.append(kw)
    print keywords

#    model = PositionDependentVectorModel(keywords, winSize=args.win,
#                                         wdim=args.dim, stepsize=args.lr,
#                                         reg=args.reg,
#                                         batchsize=args.batch)
#    model = ConstantAttentionVectorModel(keywords, winSize=args.win,
#                                         wdim=args.dim, stepsize=args.lr,
#                                         reg=args.reg,
#                                         batchsize=args.batch)
    model = NonLinearVectorModel(keywords, winSize=args.win,
                                 wdim=args.dim, zdim=args.zdim,
                                 stepsize=args.lr,
                                 reg=args.reg,
                                 batchsize=args.batch)
    if args.restore is not None:
        model.restoreFrom(args.restore)
        print 'Restored model from %s' % args.restore

    if args.command == 'test':
        model.test(filesAndTokens)

    elif args.command == 'predict':
        for _ in range(100):
            randomFile = random.choice(fileSubset)
            outputFile = os.path.basename(randomFile) + '.html'
            print 'Evaluating on %s' % randomFile
            tokens, content = utils.tokenize(randomFile, True)
            annotations = model.testOverlap(tokens)
            spans = utils.matchTokensToContent(tokens, content)
            assert len(spans) == len(tokens)
            contentAnns = [-1 for _ in range(len(content))]
            for ann,span in zip(annotations,spans):
                for ia,a in enumerate(ann):
                    contentAnns[ia+span[0]] = a

            print 'Generating %s' % outputFile

            with open(outputFile, 'w') as fp:
                print >> fp, utils.HTMLHeader()
                print >> fp, utils.colorizedHTML(content, contentAnns)
                print >> fp, utils.HTMLFooter()

    else:
        model.train(filesAndTokens, ckpt_prefix=args.ckpt)
