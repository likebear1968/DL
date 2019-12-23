import numpy as np
from layers.layerBase import LayerBase

class Affine(LayerBase):
    def __init__(self, w, b):
        super().__init__(params=[w, b], grads=[np.zeros_like(w), np.zeros_like(b)])
        self.x, self.x_shape = None, None

    def forward(self, x, train=True):
        self.x_shape = np.shape(x)
        self.x = np.reshape(x, (np.shape(x)[0], -1))
        return np.dot(self.x, self.params[0]) + self.params[1]

    def backward(self, dout=1):
        self.grads[0][...] = np.dot(self.x.T, dout)
        self.grads[1][...] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params[0].T)
        return np.reshape(dx, self.x_shape)
