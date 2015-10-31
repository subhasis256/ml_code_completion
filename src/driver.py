import fnmatch as fn
import os
import re
from vectorModel import PositionDependentVectorModel
from utils import matchingFiles, tokenize

if __name__ == '__main__':
    import cPickle as pkl
    import sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('--restore',
                        help='file from which to restore parameters for testing')
    parser.add_argument('--ckpt',
                        default='',
                        help='prefix of checkpoint files')
    parser.add_argument('--win', type=int,
                        default=40,
                        help='window size')
    parser.add_argument('--dim', type=int,
                        default=32,
                        help='word vector dimensions')
    parser.add_argument('--keywords',
                        default='../key_words/c',
                        help='file containing key words and their frequencies')
    parser.add_argument('--lr',
                        default=0.05,
                        help='learning rate')
    parser.add_argument('--batch',
                        default=32,
                        help='batch size')
    args = parser.parse_args()

    files = matchingFiles(['../data/linux'], ['c', 'h'])
    filesAndTokens = []

    if args.command == 'train':
        fileSubset = files[:20000]
    else:
        fileSubset = files[20000:22000]

    for i,name in enumerate(fileSubset):
        if i % 1000 == 0:
            print '%d files done' % i
        filesAndTokens.append((name,tokenize(name)))
    print len(filesAndTokens)
    print sum([len(tokens) for name,tokens in filesAndTokens])

    keywords = []
    with open(args.keywords) as fp:
        for line in fp:
            kw = re.sub(' [0-9]*$', '', line.strip())
            keywords.append(kw)
    print keywords

    model = PositionDependentVectorModel(keywords, winSize=args.win,
                                         wdim=args.dim, stepsize=args.lr,
                                         batchsize=args.batch)
    if args.command == 'test':
        model.restoreFrom(args.restore)
        model.test(filesAndTokens)
    else:
        model.train(filesAndTokens, ckpt_prefix=args.ckpt)
#    model.multiPredict([34532, 7, 99142, 93, 99142, 3, 105902, 2, 98774, 7, 233,
#                        4, 103368, 2, 13, 1, 99142, 3, 105902, 7, 93610, 11,
#                        99142, 3, 105902, 7, 93610, 27, 98774, 0, 99142, 3,
#                        105902, 7, 100179, 5, 59, 2, 12, 195])
#    model.multiPredict([4, 81, 2, 370, 3, 63638, 4, 43289, 1, 202, 5, 44487, 1,
#                        370, 3, 44481, 3, 108, 3, 23, 0, 370, 3, 68940, 0, 370,
#                        3, 65951, 0, 41742, 5, 43292, 1, 370, 3, 44481, 5, 127,
#                        21, 282])
#    model.multiPredict([26547, 1, 8, 26493, 9, 6429, 5, 26, 26548, 1, 8, 26493,
#                        9, 6429, 5, 26, 26525, 1, 8, 26493, 9, 6429, 0, 16,
#                        26512, 5, 26, 26551, 1, 8, 26493, 9, 6429, 5, 26499,
#                        26552, 1, 8, 26493, 9])
#    model.multiPredict([3, 589, 2, 691, 3, 6060, 7, 159877, 4, 159878, 2,
#                        158333, 3, 158351, 4, 154357, 107, 158333, 3, 159703, 4,
#                        491, 2, 158333, 3, 6127, 547, 103, 4, 159870, 27, 691,
#                        3, 6060, 0, 159887, 5, 164, 1, 103])
#    model.multiPredict([158592, 55, 824, 5, 12, 160, 51, 685, 27, 824, 85, 108,
#                        4, 6390, 27, 824, 0, 8, 158290, 0, 158583, 5, 282, 27,
#                        158554, 3, 134, 0, 69, 5, 103, 4, 158636, 1, 108, 5,
#                        332, 27, 158554, 3])
#    model.multiPredict([3, 37265, 5, 12, 410, 27, 250, 3, 134, 5, 12, 26, 37048,
#                        1, 8, 36866, 9, 36867, 0, 8, 36901, 9, 215, 11, 15, 8,
#                        37135, 9, 250, 77, 8, 37135, 116, 215, 3, 15254, 19, 14,
#                        74, 348])
