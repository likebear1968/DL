import numpy as np
from layers.layerBase import LayerBase

class Softmax(LayerBase):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x, train=True):
        self.y = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.y /= np.sum(self.y, axis=-1, keepdims=True)
        return self.y

    def backward(self, dout=1):
        dx = self.y * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.y * sumdx
        return dx
