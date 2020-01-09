import numpy as np
from layers.layerBase import LayerBase

class LeakyRelu(LayerBase):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

  def forward(self, x, train=True):
    return np.where(x > 0, x , self.alpha * x)

  def backward(self, dout=1):
    return np.where(x > 0, 1, self.alpha)
