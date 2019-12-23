import numpy as np
from layers.layerBase import LayerBase

class Sigmoid(LayerBase):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x, train=True):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout=1):
        return dout * (1.0 - self.y) * self.y
