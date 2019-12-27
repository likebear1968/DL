import numpy as np
from layers.layerBase import LayerBase

def func(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    y /= np.sum(y, axis=-1, keepdims=True)
    return y

class Softmax(LayerBase):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x, train=True):
        self.y = func(x)
        return self.y

    def backward(self, dout=1):
        dx = self.y * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.y * sumdx
        return dx
