"""
Precomputed Box Kernel
"""
import thread
import numpy as np

class EmpiricalPrecomptuedBoxKernel(object):

    def __init__(self, idxfile, kernelfile,
            dataset, ktype, epsilon, delta, seed,
            p, empirical_labels):

        with open(idxfile, 'r') as f:
            self.idx = dict([(key, i)
                for i, key in enumerate([line.strip() for line in f])])

        self.kernel_path = kernelfile
        self.connection_thread = None
        self._connection = None

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT dataset_id FROM datasets '
            'WHERE dataset_name=?', (dataset,)
        )
        self.dataset_id = cursor.fetchone()[0]

        cursor.execute(
            'SELECT ktype_id FROM ktypes '
            'WHERE ktype_name=?', (ktype,)
        )
        self.ktype_id = cursor.fetchone()[0]

        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.p = p
        self.empirical_labels = [self.idx[l] for l in empirical_labels]

        self._cached_kernel = None

    def get_connection(self):
        current_thread = thread.get_ident()
        if (self._connection is None
            or self.connection_thread != current_thread):
            import sqlite3
            self.connection_thread = current_thread
            self._connection = sqlite3.connect(self.kernel_path)
        return self._connection

    def _get_cached_kernel(self):
        if self._cached_kernel is None:
            self._cached_kernel = dict()
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(
                'SELECT i, j, mantissa, exponent FROM kernel '
                'WHERE dataset_id=? AND ktype_id=? AND '
                'epsilon=? AND delta=? '
                'AND seed=?',
                (self.dataset_id, self.ktype_id, self.epsilon, self.delta, self.seed)
            )
            for i, j, m, e in cursor.fetchall():
                v = np.power(m, 1.0/self.p)*np.power(10, (e / self.p))
                if v is np.inf:
                    raise ValueError('Overflow in kernel value (%d, %d) !' % (i, j))
                self._cached_kernel[i, j] = v
                if i != j: self._cached_kernel[j, i] = v

        return self._cached_kernel

    def feature_map(self, labels):
        labels = [self.idx[str(l)] for l in labels]
        K = self._get_cached_kernel()
        X = np.array([[K[l, e] for e in self.empirical_labels]
                      for l in labels])
        return X
