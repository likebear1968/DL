import numpy as np
from layers.layerBase import LayerBase

class Dropout(LayerBase):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.mask = None

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x * self.mask
        return x * (1.0 - self.ratio)

    def backward(self, dout=1):
        return dout * self.mask
