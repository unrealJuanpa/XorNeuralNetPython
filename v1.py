import numpy as np

# busca en el numero de iteraciones correctas (implementado)
# busca las operaciones a realizar (AND, OR, NAND, NOR, XOR, XNOR) (no implementado)

class XorModel(object):
    def __init__(self, ninputs:int, noutputs: int) -> None:
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.bestiters = np.zeros(shape=(noutputs,), dtype=int)
        self.bestacc = np.zeros(shape=(noutputs,), dtype=np.float32)

    def op(self, x):
        return np.logical_xor(x[:, -3], x[:, -1])

    def compute(self, x):
        return np.concatenate([self.op(x)[..., np.newaxis], x[:, :-1]], axis=-1, dtype=np.uint8)

    def fit(self, X, Y, maxiters):
        assert X.shape[0] == Y.shape[0]
        # [None, ninputs]
        # [None, noutputs]

        for i in range(1, maxiters+1, 1):
            X = self.compute(X)

            for j in range(self.noutputs):
                actualop = np.sum(X[:, -1] == Y[:, j])/X.shape[0]

                if actualop > self.bestacc[j]:
                    self.bestacc[j] = actualop
                    self.bestiters[j] = i

            kl = np.mean(self.bestacc)
            print(f"Iter {i}/{maxiters} \t Acc: {kl} \t Desv: {np.std(self.bestacc)}")
            if kl == 1: break


X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
], dtype=np.bool_)

Y = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0]
], dtype=np.bool_)

m = XorModel(ninputs=X.shape[1], noutputs=Y.shape[1])
m.fit(X, Y, maxiters=300000)