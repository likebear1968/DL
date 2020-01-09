import numpy as np
from layers.layerBase import LayerBase

class Tanh(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, train=True):
        return (np.exp(x) – np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def backward(self, dout=1):
        return 4 / (np.exp(dout) + np.exp(-dout)) ** 2
