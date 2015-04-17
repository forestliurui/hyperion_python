"""
Implements various bag-distance metrics
"""
import numpy as np

class EMD(object):

    def __init__(self):
        from emd import emd
        self.emd = emd

    def __call__(self, B, C):
        return np.array([[self.emd(X, Y) for Y in C]
                         for X in B])

class KernelEMD(object):

    def __init__(self, *args, **kwargs):
        from emd import emd
        if 'kernel' not in kwargs:
            raise ValueError('Kernel function must be supplied for KernelEMD.')
        self.kernel = kwargs.pop('kernel')
        self.emd = emd

    def d(self, X, Y):
        xx = (x.reshape((1, -1)) for x in X)
        Kxx = np.array([self.kernel(x, x) for x in xx])
        yy = (y.reshape((1, -1)) for y in Y)
        Kyy = np.array([self.kernel(y, y) for y in yy])
        Kxy = self.kernel(X, Y)
        D = np.sqrt(np.outer(Kxx, np.ones(len(Y)))
                  + np.outer(np.ones(len(X)), Kyy)
                  - 2*Kxy)
        return D

    def __call__(self, B, C):
        return np.array([[self.emd(None, None,
                            distance='precomputed', D=self.d(X, Y))
                          for Y in C] for X in B])
