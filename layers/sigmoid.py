import numpy as np
from layers.layerBase import LayerBase

def func(x):
    return 1 / (1 + np.exp(-np.array(x)))

class Sigmoid(LayerBase):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x, train=True):
        self.y = func(x)
        return self.y

    def backward(self, dout=1):
        return dout * (1.0 - self.y) * self.y
