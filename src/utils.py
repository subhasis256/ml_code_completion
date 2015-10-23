import numpy as np

def softmaxLossAndGrads(scores, tgts):
    deltas = scores - np.amax(scores, axis=1, keepdims=True)
    edeltas = np.exp(deltas)
    probs = edeltas/np.sum(edeltas, axis=1, keepdims=True)
    grads = probs.copy()
    nb = scores.shape[0]
    grads[np.arange(nb),tgts] -= 1
    loss = -np.mean(np.log(probs[np.arange(nb),tgts]+1e-13))
    return loss, grads
