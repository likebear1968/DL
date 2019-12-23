from layers.layerBase import LayerBase

class Relu(LayerBase):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x, train=True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
