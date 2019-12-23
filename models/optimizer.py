import numpy as np
from abc import ABCMeta, abstractmethod

class OptimizerBase(metaclass=ABCMeta):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.m, self.v = None, None
        self.eta = 1e-7

    @abstractmethod
    def update(self, params, grads):
        pass

class SGD(OptimizerBase):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum(OptimizerBase):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

class AdaGrad(OptimizerBase):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.v[i]) + self.eta)

class RMSprop(OptimizerBase):
    def __init__(self, lr=0.01, decay_rate = 0.99):
        super().__init__(lr)
        self.decay_rate = decay_rate

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] *= self.decay_rate
            self.v[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.v[i]) + self.eta)

class Adam(OptimizerBase):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        super().__init__(lr)
        self.beta1, self.beta2 = beta1, beta2
        self.iter = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        self.iter += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.iter) / (1 - self.beta1 ** self.iter)
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eta)
