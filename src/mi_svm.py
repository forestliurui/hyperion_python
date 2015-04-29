"""
Implements a bag-kernel based SVM using the
scikit-learn SVM implementation
"""
import numpy as np
from sklearn.svm import SVC, NuSVR

import kernel

MAX_ITERS = 100000

class SVM(object):

    def __init__(self, **parameters):
        svm_params = {'kernel' : 'precomputed'}
        if 'C' in parameters:
            svm_params['C'] = parameters.pop('C')
        self.estimator = SVC(**svm_params)

        # Get kernel name and pass remaining parameters to kernel
        kernel_name = parameters.pop('kernel')
        self.kernel = kernel.by_name(kernel_name, **parameters)

    def fit(self, X, y, weights=None):
        X = np.asarray(X)
        self.fit_data = X
        #  X is a list of arrays so applying asarray function to everything in that list
        #  If you passed in a list of lists, if each bag is an array the asarray funciton just returns it
        #  but it converts a list of lists to a numpy array.
        #import pdb; pdb.set_trace()
	self.gram_matrix = self.kernel(X, X)
        self.estimator.fit(self.gram_matrix, y, sample_weight = weights)
        return self

    def predict(self, X=None):
        if X is None:
            gram_matrix = self.gram_matrix
        else:
            X = np.asarray(X)
            gram_matrix = self.kernel(X, self.fit_data)
        return self.estimator.decision_function(gram_matrix)

class MIKernelSVM(object):

    def __init__(self, **parameters):
        svm_params = {'kernel' : 'precomputed'}
        if 'C' in parameters:
            svm_params['C'] = parameters.pop('C')
        self.estimator = SVC(**svm_params)

        # Get kernel name and pass remaining parameters to kernel
        mi_kernel_name = parameters.pop('kernel')
        self.mi_kernel = kernel.by_name(mi_kernel_name, **parameters)

    def fit(self, X, y):
        X = map(np.asarray, X)
        self.fit_data = X
        self.gram_matrix = self.mi_kernel(X, X)
        self.estimator.fit(self.gram_matrix, y)
        return self

    def predict(self, X=None):
        if X is None:
            gram_matrix = self.gram_matrix
        else:
            X = map(np.asarray, X)
            gram_matrix = self.mi_kernel(X, self.fit_data)
        return self.estimator.decision_function(gram_matrix)

class MIKernelSVR(MIKernelSVM):

    def __init__(self, **parameters):
        svr_params = {
            'kernel' : 'precomputed',
            'max_iter': MAX_ITERS,
        }
        if 'C' in parameters:
            svr_params['C'] = parameters.pop('C')
        if 'nu' in parameters:
            svr_params['nu'] = parameters.pop('nu')
        self.estimator = NuSVR(**svr_params)

        # Get kernel name and pass remaining parameters to kernel
        mi_kernel_name = parameters.pop('kernel')
        self.mi_kernel = kernel.by_name(mi_kernel_name, **parameters)

    def fit(self, X, y):
        X = map(np.asarray, X)
        self.fit_data = X
        self.gram_matrix = self.mi_kernel(X, X)
        self.estimator.fit(self.gram_matrix, y)
        return self

    def predict(self, X=None):
        if X is None:
            gram_matrix = self.gram_matrix
        else:
            X = map(np.asarray, X)
            gram_matrix = self.mi_kernel(X, self.fit_data)
        return self.estimator.predict(gram_matrix)
