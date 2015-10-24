import numpy as np
import sys

def softmaxLossAndGrads(scores, tgts):
    deltas = scores - np.amax(scores, axis=1, keepdims=True)
    edeltas = np.exp(deltas)
    probs = edeltas/np.sum(edeltas, axis=1, keepdims=True)
    grads = probs.copy()
    nb = scores.shape[0]
    grads[np.arange(nb),tgts] -= 1
    loss = -np.mean(np.log(probs[np.arange(nb),tgts]+1e-13))
    return loss, grads

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
