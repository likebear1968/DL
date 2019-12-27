import numpy as np

class Model:
    def __init__(self):
        self.params, self.grads, self.layers = [], [], []

    def append(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.grads += layer.grads

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers[0:-1]):
            dout = layer.backward(dout)
        return dout

    def summary(self, dummy):
        #y = np.zeros(np.shape(self.layers[0].params[0].T))
        #y = np.zeros_like(self.layers[0].params[0]).T
        print('-' * 50)
        for layer in self.layers:
            print('layer :', layer.__class__.__name__)
            for param in layer.params:
                print('param :', param.shape)
            dummy = layer.forward(dummy)
            print('output:', dummy.shape)
        print('-' * 50)
