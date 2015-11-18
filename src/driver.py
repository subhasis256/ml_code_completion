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

    def listparse(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        choices=['train', 'test', 'predict'],
                        help='train/test/predict')
    parser.add_argument('-p', '--project',
                        default='linux',
                        help='project name, should be a subdir in data')
    parser.add_argument('-l', '--langs',
                        type=listparse,
                        default=['c', 'h'],
                        help='languages to use as a comma separated list')
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

    data_dir = os.path.join('../data', args.project)
    files = utils.matchingFiles([data_dir], args.langs)
    filesAndTokens = []

    # choose the first half of files based on a deterministic random range
    robj = random.Random(12345)
    robj.shuffle(files)

    if args.command == 'train':
        fileSubset = files[:len(files)/2]
    elif args.command == 'test':
        fileSubset = files[len(files)/2:]
    else:
        fileSubset = files[len(files)/2:]

    if args.command == 'train' or args.command == 'test':
        for i,name in enumerate(fileSubset):
            if i % 1000 == 0:
                print '%d files done' % i
            filesAndTokens.append((name,utils.tokenize(name)))
        print len(filesAndTokens)
        print sum([len(tokens) for name,tokens in filesAndTokens])

    keywords = []
    keywords_file = os.path.join('../key_words', args.project)
    with open(keywords_file) as fp:
        for line in fp:
            kw = re.sub(' [0-9]*$', '', line.strip())
            keywords.append(kw)
    print keywords

#    model = PositionDependentVectorModel(keywords, winSize=args.win,
#                                         wdim=args.dim, stepsize=args.lr,
#                                         reg=args.reg)
#    model = ConstantAttentionVectorModel(keywords, winSize=args.win,
#                                         wdim=args.dim, stepsize=args.lr,
#                                         reg=args.reg)
    model = NonLinearVectorModel(keywords, winSize=args.win,
                                 wdim=args.dim, zdim=args.zdim,
                                 stepsize=args.lr,
                                 reg=args.reg)
    if args.restore is not None:
        model.restoreFrom(args.restore)
        print 'Restored model from %s' % args.restore

    if args.command == 'test':
        model.test(filesAndTokens, batchsize=args.batch)

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
        model.train(filesAndTokens, ckpt_prefix=args.ckpt, batchsize=args.batch)
