import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss, metrics=[]):
        self.model, self.optimizer, self.loss, self.metrics = model, optimizer, loss, metrics

    def fit(self, x, t, epoch_size, batch_size):
        size = len(x)
        iters = size // batch_size
        for _ in range(epoch_size):
            mt = {}
            for _ in range(iters):
                mask = np.random.choice(size, batch_size)
                score = self.model.forward(x[mask], True)
                lss, dout, obs, prd = self.loss(t[mask], score)
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)
                self.add_metrics(mt, lss, obs, prd)
            self.print_metrics(mt, iters)

    def predict(self, x, t):
        mt = {}
        iters = len(x)
        for i in range(iters):
            score = self.model.forward(x[[i]])
            lss, _, obs, prd = self.loss(t[i], score)
            self.add_metrics(mt, lss, obs, prd)
        self.print_metrics(mt, iters)

    def add_metrics(self, mt, lss, obs, prd):
        mt['loss'] = mt.get('loss', 0) + lss
        for m in np.append([], self.metrics):
            nm = m.func.__name__
            mt[nm] = mt.get(nm, 0) + m.metrics(obs, prd)
        return mt

    def print_metrics(self, mt, cnt):
        print(*[f'{k.ljust(10)}:{v / cnt:.5f} ' for k, v in mt.items()], sep='|')
