import numpy as np
import sys
import re
import fnmatch as fn
import os

def matchingFiles(data_dirs, suffixes):
    globs = map(lambda x : '*.' + x, suffixes)
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for glob in globs:
                for filename in fn.filter(filenames, glob):
                    matches.append(os.path.join(root, filename))
    return matches

def tokenize(fileName):
    allTokens = []
    with open(fileName) as data:
        content = re.sub(r'/\*.*?\*/', '', data.read(),
                         flags=re.MULTILINE|re.DOTALL)
        for line in content.split('\n'):
            allTokens += [token.strip()
                          for token in re.split('(\W+)', line)
                          if len(token.strip()) > 0]
    return allTokens

def softmaxLossAndGrads(scores, tgts):
    deltas = scores - np.amax(scores, axis=1, keepdims=True)
    edeltas = np.exp(deltas)
    probs = edeltas/np.sum(edeltas, axis=1, keepdims=True)
    grads = probs.copy()
    nb = scores.shape[0]
    grads[np.arange(nb),tgts] -= 1
    loss = -np.mean(np.log(probs[np.arange(nb),tgts]+1e-13))
    return loss, probs, grads

def colorPrint(*args, **kwargs):
    cmap = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'black': '\033[30m',
    }
    color = kwargs['color'] if 'color' in kwargs else 'black'
    if sys.stdout.isatty():
        sys.stdout.write(cmap[color])
        for arg in args:
            sys.stdout.write(str(arg))
            sys.stdout.write(' ')
        sys.stdout.write('\033[00m')
    else:
        for arg in args:
            sys.stdout.write(str(arg))
