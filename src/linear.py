"""
Implements Linear L1-SVM
"""
import numpy as np
from cvxopt import matrix as cvxmat, spmatrix, sparse
from cvxopt.solvers import lp, options

import kernel as klib

class L1SVM(object):

    def __init__(self, **parameters):
        self.C = parameters.pop('C', 1.0)
        self.scale_C = parameters.pop('scale_C', True)
        self.verbose = parameters.pop('verbose', True)

        kernel_name = parameters.pop('kernel')
        self.kernel = klib.by_name(kernel_name, **parameters)

        self.data = None
        self.gram_matrix = None
        self.w = None

    def fit(self, X, y):
        y = (2.0*(np.asarray(y) == 1) - 1.0)
        X = np.asarray(X)
        self.data = X
        self.gram_matrix = self.kernel(X, X)

        n = len(X)
        if self.scale_C:
            C = self.C / float(n)
        else:
            C = self.C

        Y = np.diag(y)
        YK = sparse(cvxmat(np.dot(Y, self.gram_matrix)))
        Y1 = sparse(cvxmat(np.dot(Y, np.ones((n, 1)))))
        YI = sparse(cvxmat(np.dot(Y, np.eye(n))))

        c = cvxmat(sparse([spo(n), spz(n + 1), spo(n, v=C)]))
        G = sparse(t([[spz(n, n),  YK,        Y1,     YI       ],
                      [spz(n, n),  spz(n, n), spz(n), spI(n)   ],
                      [spI(n),    -spI(n),    spz(n), spz(n, n)],
                      [spI(n),     spI(n),    spz(n), spz(n, n)],
                      [spI(n),     spz(n, n), spz(n), spz(n, n)]]))
        h = cvxmat(sparse([spo(n), spz(4*n)]))

        # We want Gx >= h, but solver uses Gx <= h, so negate:
        xstar, _ = linprog(c, -G, -h, verbose=self.verbose)

        try:
            self.w = xstar[n:2*n + 1].reshape((-1,))
        except ValueError as e:
            print e
            self.w = np.zeros(n + 1)

    def predict(self, X=None):
        if self.w is None:
            raise Exception('`fit` must be called before `predict`')

        if X is None:
            n = len(self.gram_matrix)
            gram_matrix = self.gram_matrix
        else:
            n = len(X)
            gram_matrix = self.kernel(X, self.data)

        K = np.hstack([gram_matrix, np.ones((n, 1))])
        return np.dot(K, self.w).reshape((-1,))

def spI(n):
    """Create a sparse identity matrix"""
    r = range(n)
    return spmatrix(1.0, r, r)

def spz(r, c=1):
    """Create a sparse zero vector or matrix"""
    return spmatrix([], [], [], (r, c))

def spo(r, v=1.0):
    """Create a sparse one vector"""
    return spmatrix(v, range(r), r*[0])

def t(list_of_lists):
    """
    Transpose a list of lists, since 'sparse'
    takes arguments in column-major order,
    which is stupid in my opinion.
    """
    return map(list, zip(*list_of_lists))

def linprog(*args, **kwargs):
    """
    min c^T*x
    s.t. Gx <= h
         Ax = b
    args: c, G, h, A, b
    """
    verbose = kwargs.get('verbose', False)
    # Save settings and set verbosity
    old_settings = _apply_options({'show_progress': verbose,
                                   'LPX_K_MSGLEV': int(verbose)})

    # Optimize
    results = lp(*args, solver='glpk')

    # Restore settings
    _apply_options(old_settings)

    # Check return status
    status = results['status']
    if not status == 'optimal':
        from sys import stderr
        print >> stderr, ('Warning: termination of lp with status: %s'
                          % status)

    # Convert back to NumPy array
    # and return solution
    xstar = np.array(results['x'])
    return xstar, results['primal objective']

def _apply_options(option_dict):
    old_settings = {}
    for k, v in option_dict.items():
        old_settings[k] = options.get(k, None)
        if v is None:
            del options[k]
        else:
            options[k] = v
    return old_settings
