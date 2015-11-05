import numpy as np
import sys
import re
import fnmatch as fn
import os
import cgi

def matchingFiles(data_dirs, suffixes):
    globs = map(lambda x : '*.' + x, suffixes)
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for glob in globs:
                for filename in fn.filter(filenames, glob):
                    matches.append(os.path.join(root, filename))
    return matches

def tokenize(fileName, retcontent=False):
    allTokens = []
    with open(fileName) as data:
        content = re.sub(r'/\*.*?\*/', '', data.read(),
                         flags=re.MULTILINE|re.DOTALL)

        for line in content.split('\n'):
            allTokens += [token.strip()
                          for token in re.split(r'(\W+)', line)
                          if len(token.strip()) > 0]
    if not retcontent:
        return allTokens
    else:
        return allTokens, content

def matchTokensToContent(tokens, content):
    cid = 0
    spans = []
    for tid,tok in enumerate(tokens):
        while content[cid] != tok[0]:
            cid += 1
        spans.append((cid, cid+len(tok)))
        cid += len(tok)
    return spans

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
        for ii,arg in enumerate(args):
            sys.stdout.write(str(arg))
            if ii != len(args)-1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[00m')
    else:
        for arg in args:
            sys.stdout.write(str(arg))


def HTMLHeader():
    return """<!DOCTYPE html>
<html>
<head>
<style>
span.c {
    background-color: #CCFFCC;
}
span.pc {
    background-color: #FFEEBB;
}
span.w {
    background-color: #FFCCCC;
}
</style>
</head>
<body>
<pre>"""

def HTMLFooter():
    return """</pre>
</body>
</html>
"""

def spanStart(annotation):
    annClass = {0: 'w', 1: 'pc', 2: 'c'}
    if annotation in annClass:
        return "<span class=\"" + annClass[annotation] + "\">"
    else:
        return ""

def spanEnd(annotation):
    annClass = {0: 'w', 1: 'pc', 2: 'c'}
    if annotation in annClass:
        return "</span>"
    else:
        return ""

def colorizedHTML(content, annotations):
    """
    annotation convention:
    0 -> wrong
    1 -> partially correct
    2 -> correct
    -1 -> no annotation
    """
    pa = None
    s = ''
    for c,a in zip(content, annotations):
        if a != pa:
            # start a new span
            s += spanEnd(pa)
            s += spanStart(a)
        s += cgi.escape(c)
        pa = a
    s += spanEnd(pa)
    return s


def softmaxLossAndGrads(scores, tgts):
    deltas = scores - np.amax(scores, axis=1, keepdims=True)
    edeltas = np.exp(deltas)
    probs = edeltas/np.sum(edeltas, axis=1, keepdims=True)
    grads = probs.copy()
    nb = scores.shape[0]
    grads[np.arange(nb),tgts] -= 1
    loss = -np.mean(np.log(probs[np.arange(nb),tgts]+1e-13))
    return loss, probs, grads

def relu(x):
    return 0.5*(x + np.abs(x))

def drelu(dy, y):
    return dy*(y > 0)
