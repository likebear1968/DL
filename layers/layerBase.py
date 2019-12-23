class LayerBase():
    def __init__(self, params=[], grads=[]):
        self.params, self.grads = params, grads

    def forward(self, x, train=True):
        return x

    def backward(self, dout=1):
        return dout
