import numpy as np

class SGDOptimizer:
    def __init__(self, step):
        self.step = step

    def update(self, param, grad):
        param -= self.step * grad

class MomentumOptimizer:
    def __init__(self, step, momentum):
        self.cache = {}
        self.step = step
        self.momentum = momentum

    def update(self, param, grad):
        if id(param) not in self.cache:
            self.cache[id(param)] = np.zeros_like(grad)
        vp = self.cache[id(param)]
        v = self.momentum * vp - self.step * grad
        param += v
        self.cache[id(param)] = v

class AdagradOptimizer:
    def __init__(self, step):
        self.ssq = {}
        self.step = step

    def update(self, param, grad):
        if id(param) not in self.ssq:
            self.ssq[id(param)] = np.zeros_like(grad)

        self.ssq[id(param)] += grad * grad
        param -= self.step * grad / (np.sqrt(self.ssq[id(param)]) + 1e-14)
