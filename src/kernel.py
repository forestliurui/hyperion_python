"""
Implement the various kernel functions
"""
import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.stats import scoreatpercentile

import distances
import progress


try:
    from boxkernel import BoxKernel
except ImportError:
    BoxKernel = None

class OracleKernel(object):

    def __init__(self, **parameters):
        parameters = dict(**parameters)
        if 'base_kernel' not in parameters:
            raise ValueError('A base kernel name must be supplied.')
        base_kernel_name = parameters.pop('base_kernel')
        # Remaining parameters go to base kernel
        self.base_kernel = by_name(base_kernel_name, **parameters)

    def __call__(self, B, C):

        if not all(len(X) == 1 for X in B) or not all(len(Y) == 1 for Y in C):
            raise ValueError('Oracle Kernel can only be used for singleton bags')

        X = np.vstack(B)
        Y = np.vstack(C)

        return self.base_kernel(X, Y)

class NormalizedSetKernel(object):

    def __init__(self, **parameters):
        parameters = dict(**parameters)
        if 'base_kernel' not in parameters:
            raise ValueError('A base kernel name must be supplied.')
        base_kernel_name = parameters.pop('base_kernel')
        self.normalization = parameters.pop('normalization', 'none')
        # Remaining parameters go to base kernel
        self.base_kernel = by_name(base_kernel_name, **parameters)

    def __call__(self, B, C):

        # To monitor progress of big kernel computations
        total = len(B)*len(C)
        if total > 1:
            prog = progress.ProgressMonitor(total=total,
                                            print_interval=10,
                                            msg='Computing kernel')
            def status(x):
                prog.increment()
                return x
        else:
            status = lambda x: x

        K = np.array([[status(np.sum(self.base_kernel(X, Y))) for Y in C]
                      for X in B])
        if self.normalization == 'none':
            norms = np.ones(K.shape)

        elif self.normalization == 'averaging':
            normB = np.array([len(X) for X in B], dtype=float)
            normC = np.array([len(Y) for Y in C], dtype=float)
            norms = np.outer(normB, normC)

        elif self.normalization == 'featurespace':
            normB = np.array([np.sqrt(np.sum(self.base_kernel(X, X)))
                              for X in B], dtype=float)
            normC = np.array([np.sqrt(np.sum(self.base_kernel(Y, Y)))
                              for Y in C], dtype=float)
            norms = np.outer(normB, normC)

        else:
            raise ValueError('Unknown normalization "%s"' % self.normalization)

        return (K / norms)

class StatisticKernel(object):

    def __init__(self, **parameters):
        parameters = dict(**parameters)
        if 'base_kernel' not in parameters:
            raise ValueError('A base kernel name must be supplied.')
        base_kernel_name = parameters.pop('base_kernel')
        # Remaining parameters go to base kernel
        self.base_kernel = by_name(base_kernel_name, **parameters)

    def __call__(self, B, C):
        # Construct explicit feature vectors from bag statsitics
        X = np.vstack([np.hstack([np.min(bag, axis=0), np.max(bag, axis=0)])
                       for bag in B])
        Y = np.vstack([np.hstack([np.min(bag, axis=0), np.max(bag, axis=0)])
                       for bag in C])
        return self.base_kernel(X, Y)

class LinearKernel(object):

    def __init__(self, **parameters): pass

    def __call__(self, X, Y):
        return np.dot(X, Y.T)

class PolynomialKernel(object):

    def __init__(self, **parameters):
        if 'power' not in parameters:
            raise ValueError('Power must be specified for polynomial kernel.')
        self.p = parameters['power']

    def __call__(self, X, Y):
        return np.power(1.0 + np.dot(X, Y.T), self.p)

class QuadraticKernel(PolynomialKernel):

    def __init__(self, **parameters):
        super(QuadraticKernel, self).__init__(power=2)

class RBFKernel(object):

    def __init__(self, **parameters):
        if 'gamma' not in parameters:
            raise ValueError('Gamma must be specified for RBF kernel.')
        self.gamma = parameters['gamma']

    def __call__(self, X, Y):
        return np.exp(-self.gamma*cdist(X, Y, 'sqeuclidean'))

class TwoLevelSetKernel(object):

    def __init__(self, **parameters):
        parameters = dict(**parameters)

        # Get second-level kernel (must be RBF for now) and parameters
        if 'second_kernel' not in parameters:
            raise ValueError('A second kernel name must be supplied.')
        self.second_kernel = parameters.pop('second_kernel')
        if self.second_kernel == 'rbf':
            if 'gamma2' not in parameters:
                raise ValueError('"gamma2" must be specified for two-level RBF kernel.')
            self.gamma2 = parameters.pop('gamma2')

        else:
            raise ValueError('Right now the second kernel can only be RBF')

        # Force averaging-normalized set kernel
        parameters['normalization'] = 'averaging'
        self.set_kernel = by_name('nsk', **parameters)

    def __call__(self, B, C): 
        diag_B = np.array([self.set_kernel([X], [X]) for X in B])
        diag_C = np.array([self.set_kernel([X], [X]) for X in C])
        K_BC = self.set_kernel(B, C)
        D = (np.outer(diag_B, np.ones(len(C)))
             + np.outer(np.ones(len(B)), diag_C) - 2*K_BC)
        return np.exp(-self.gamma2*D)

class miGraphKernel(object):

    def __init__(self, **parameters):
        if 'delta' not in parameters:
            raise ValueError('Delta must be specified for the miGraph kernel')
        self.delta = parameters.pop('delta')
        if 'base_kernel' not in parameters:
            raise ValueError('A base kernel name must be supplied.')
        base_kernel_name = parameters.pop('base_kernel')
        # Remaining parameters go to base kernel
        self.base_kernel = by_name(base_kernel_name, **parameters)

    def __call__(self, B, C):
        # Compute pairwise distances within bags
        Bdists = [cdist(bag, bag, 'euclidean') for bag in B]
        Cdists = [cdist(bag, bag, 'euclidean') for bag in C]

        # Check for avg. distance heuristic
        if self.delta <= 0:
            # Use average distance within bags
            Bdeltas = np.array([np.average(squareform(dist, checks=False))
                                for dist in Bdists])
            Cdeltas = np.array([np.average(squareform(dist, checks=False))
                                for dist in Cdists])
            # Handle bags of size 1
            Bdeltas[np.isnan(Bdeltas)] = 1.0
            Cdeltas[np.isnan(Cdeltas)] = 1.0
        else:
            Bdeltas = len(B)*[self.delta]
            Cdeltas = len(C)*[self.delta]

        #  Create affinity matrices
        v = [(dist < delta).astype(int)
             for dist, delta in zip(Bdists, Bdeltas)]
        w = [(dist < delta).astype(int)
             for dist, delta in zip(Cdists, Cdeltas)]
        
        #  Wia = 1 / sum_u^ni w_au for affinity matrix Wi.
        #  Since there are two lists of bags there are two affinity matrices, V (B) and W (C)
        #  So big_w has two indices, on for each set of Wias.
        #  big_w[0][m] is for bag m, a list of ni values, one for each instance.
        V = [1.0/np.sum(vi, axis=1) for vi in v]
        W = [1.0/np.sum(wi, axis=1) for wi in w]
        Vnorm = [np.sum(Vi) for Vi in V]
        Wnorm = [np.sum(Wi) for Wi in W]
        
        #  Now to get the kernel that's returned, K[i,j] = kG(B_i, C_j)
        K = np.zeros((len(B), len(C)))
        for i in range(len(B)):
            for j in range(len(C)):
                k = self.base_kernel(B[i], C[j])
                K[i, j] = (np.sum(np.multiply(np.outer(V[i], W[j]), k))
                           / (Vnorm[i]*Wnorm[j]))

        return K

class MIGraphKernel(object):

    def __init__(self, **parameters):
        if 'node_kernel' not in parameters:
            raise ValueError('node_kernel must be specified for the MIGraph Kernel')
        if 'edge_kernel' not in parameters:
            raise ValueError('edge_kernel must be specified for the MIGraph Kernel')
        if 'epsilon' not in parameters:
            raise ValueError('Epsilon must be specified for the MIGraph Kernel')

        # Heuristic to make computation feasible with very large bags
        self.max_edges = parameters.pop('max_edges', 0)

        node_kernel_name = parameters.pop('node_kernel')
        node_kernel_parameters = dict([(key[5:], value) for key, value in parameters.items()\
                                                        if key.startswith('node_')])
        edge_kernel_name = parameters.pop('edge_kernel')
        edge_kernel_parameters = dict([(key[5:], value) for key, value in parameters.items()\
                                                        if key.startswith('node_')])

        self.epsilon = parameters['epsilon']
        self.node_kernel = by_name(node_kernel_name, **node_kernel_parameters)
        self.edge_kernel = by_name(edge_kernel_name, **edge_kernel_parameters)

    def __call__(self, B, C):

        # Precompute edge features for efficiency
        edge_B = [self.edge_features(X) for X in B]
        edge_C = [self.edge_features(Y) for Y in C]

        K = np.array([[self.k(X, edge_X, Y, edge_Y)
                       for Y, edge_Y in zip(C, edge_C)]
                      for X, edge_X in zip(B, edge_B)])

        # Normalization
        normB = np.sqrt([self.k(X, edge_X, X, edge_X)
                         for X, edge_X in zip(B, edge_B)])
        normC = np.sqrt([self.k(Y, edge_Y, Y, edge_Y)
                         for Y, edge_Y in zip(C, edge_C)])
        return (K / np.outer(normB, normC))

    def k(self, X, edge_X, Y, edge_Y):
        knodes = np.sum(self.node_kernel(X, Y))
        kedges = np.sum(self.edge_kernel(edge_X, edge_Y))
        return knodes + kedges

    def edge_features(self, X):
        if len(X) <= 1: return np.zeros((0, 4))
        D = cdist(X, X, 'euclidean')

        # Compute edges
        if self.epsilon <= 0:
            dists = squareform(D, checks=False)
            # Use average heuristic
            eps = np.average(dists)

            # Check if number of edges exceeds max
            pc = (100.*self.max_edges)/len(dists)
            if 0 < pc and pc < 100:
                eps2 = scoreatpercentile(dists, pc)
                if eps2 < eps:
                    # Max edges exceeded; overriding heuristic
                    eps = eps2

        else:
            eps = self.epsilon
        edges = (D < eps)
        np.fill_diagonal(edges, 0)
        n_edges = np.sum(edges) / 2

        if n_edges == 0: return np.zeros((0, 4))

        # "we use the normalized reciprocal of non-zero
        # distance as the affinity value"
        Dadj = np.array(D)
        Dadj[D == 0] = 1.0
        affinities = 1.0 / Dadj
        affinities[np.logical_not(edges)] = 0.0 # zero out non-edges
        nonzero_edges = np.logical_and(edges, D > 0)
        if np.sum(nonzero_edges) == 0:
            max_reciprocal = 1.0
        else:
            max_reciprocal = np.max(1.0 / D[nonzero_edges])
        affinities /= max_reciprocal # normalize

        degrees = np.sum(edges, axis=0)
        O = np.ones(len(X))
        dO = np.outer(degrees, O)

        # [du, dv]
        dd = np.column_stack([dO.reshape((-1,)), dO.T.reshape((-1,))])

        # [pu, pv]
        # where pu is wuv / wu*
        ww = affinities.reshape((-1,))
        w_star = np.sum(affinities, axis=0)
        w_star[w_star == 0] = 1.0
        norm = 1 / w_star
        wO = np.outer(norm, O)
        pp = np.column_stack([ww * wO.reshape((-1,)), ww * wO.T.reshape((-1,))])

        # Select only the entries for which edges exist
        # (use tril so that edges aren't double-counted)
        idx = np.nonzero(np.tril(edges).reshape((-1,)))
        V = np.hstack([dd, pp])[idx]
        return V

class BoxCountingKernel(object):

    def __init__(self, **parameters):
        if 'type' not in parameters:
            raise ValueError('type must be specified: and, or, min, and/or')
        self.type = parameters.pop('type')
        if 'gamma' not in parameters:
            raise ValueError('gamma must be specified')
        self.gamma = parameters.pop('gamma')
        if 'delta' not in parameters:
            raise ValueError('delta must be specified')
        self.delta = parameters.pop('delta')
        if 'epsilon' not in parameters:
            raise ValueError('epsilon must be specified')
        self.epsilon = parameters.pop('epsilon')

    def __call__(self, B, C):
        boxk = BoxKernel(self.gamma, self.delta, self.epsilon, self.type)
        return boxk(B, C)

class DistanceRBFKernel(object):

    def __init__(self, **parameters):
        if 'gamma' not in parameters:
            raise ValueError('Gamma must be specified for Distance RBF kernel.')
        if 'metric' not in parameters:
            raise ValueError('A "metric" must be specified for Distance RBF kernel.')
        self.gamma = parameters.pop('gamma')
        metric = parameters.pop('metric')
        if metric == 'emd':
            self.D = distances.EMD()

        elif metric == 'kemd':
            if 'emd_kernel' not in parameters:
                raise ValueError('Kernel must be specified for Kernel EMD.')
            kernel_name = parameters.pop('emd_kernel')
            kernel_parameters = dict([(key[11:], value)
                                      for key, value in parameters.items()
                                      if key.startswith('emd_kernel_')])
            kernel = by_name(kernel_name, **kernel_parameters)
            self.D = distances.KernelEMD(kernel=kernel)

        else:
            raise ValueError('Unknown distance metric "%s".' % metric)

    def __call__(self, B, C):
        return np.exp(-self.gamma*self.D(B, C))

class EmpiricalBoxKernel(object):

    def __init__(self, **parameters):
        from precomputed import EmpiricalPrecomptuedBoxKernel
        if 'idxfile' not in parameters:
            raise ValueError('Index file "idxfile" must be specified for precomptued kernel.')
        if 'kernelfile' not in parameters:
            raise ValueError('Kernel file "kernelfile" must be specified for precomputed kernel.')
        idxfile = parameters.pop('idxfile')
        kernelfile = parameters.pop('kernelfile')
        dataset = parameters.pop('dataset')
        ktype = parameters.pop('ktype')
        epsilon = parameters.pop('epsilon')
        delta = parameters.pop('delta')
        seed = parameters.pop('seed')
        p = parameters.pop('p')
        empirical_labels = parameters.pop('empirical_labels')

        self.emp = EmpiricalPrecomptuedBoxKernel(
            idxfile, kernelfile, dataset,
            ktype, epsilon, delta, seed,
            p, empirical_labels)

    def __call__(self, B, C):
        X = self.emp.feature_map(B)
        Y = self.emp.feature_map(C)
        K = np.dot(X, Y.T)
        return K / np.max(K)

_KERNELS = {
    'linear': LinearKernel,
    'poly'  : PolynomialKernel,
    'quadratic' : QuadraticKernel,
    'rbf': RBFKernel,
    'nsk' : NormalizedSetKernel,
    'stk' : StatisticKernel,
    'twolevel': TwoLevelSetKernel,
    'miGraph': miGraphKernel,
    'MIGraph' : MIGraphKernel,
    'box': BoxCountingKernel,
    'emp': EmpiricalBoxKernel,
    'distance_rbf': DistanceRBFKernel,
    'oracle': OracleKernel,
}

def by_name(kernel_name, **kernel_parameters):
    if kernel_name not in _KERNELS:
        raise ValueError('Unknown kernel "%s"' % kernel_name)
    Kernel = _KERNELS[kernel_name]
    return Kernel(**kernel_parameters)
