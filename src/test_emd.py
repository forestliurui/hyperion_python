#!/usr/bin/env python
"""
Script for testing kernel code before
running the experimental code.
"""
import numpy as np
from data import get_dataset
import time

from scipy.spatial.distance import cdist
from cvxopt import matrix as cvxmat, sparse
from linear import linprog, spI, spz
from emd import emd as tp_emd
from cv import CalcEMD2, CV_DIST_L2

def lp_emd(X, Y, X_weights=None, Y_weights=None, distance='euclidean', D=None, verbose=False):
    if distance != 'precomputed':
        n = len(X)
        m = len(Y)
        D = cdist(X, Y, distance)
        if X_weights is None:
            X_weights = np.ones(n)/n
        elif n != len(X_weights):
            raise ValueError('Size mismatch of X and X_weights')
        if Y_weights is None:
            Y_weights = np.ones(m)/m
        elif m != len(Y_weights):
            raise ValueError('Size mismatch of Y and Y_weights')
    else:
        if D is None:
            raise ValueError("D must be supplied when distance='precomputed'")
        n, m = D.shape
        if X_weights is None:
            X_weights = np.ones(n)/n
        elif n != len(X_weights):
            raise ValueError('Size mismatch of D and X_weights')
        if Y_weights is None:
            Y_weights = np.ones(m)/m
        elif m != len(Y_weights):
            raise ValueError('Size mismatch of D and Y_weights')

    vecD = D.reshape((-1, 1))

    # Set up objective function
    C = cvxmat(vecD)

    # Set up inequality constraints
    G = -spI(n*m)
    h = cvxmat(np.zeros((n*m, 1)))

    # Set up equality constraints
    Aeq = np.zeros((n+m-1, n*m))

    # Sum of rows
    for r in range(n):
        Aeq[r, r*m:(r+1)*m] = 1

    # Sum of columns
    # (Exclude final column, since contraint is determined by others)
    for c in range(m-1):
        for r in range(n):
            Aeq[c+n, r*m + c] = 1

    sparseAeq = sparse(cvxmat(Aeq))
    b = np.vstack([X_weights.reshape((-1, 1)), Y_weights[:-1].reshape((-1, 1))])
    sparseb = cvxmat(b)
    _, dist = linprog(C, G, h, sparseAeq, sparseb, verbose=verbose)
    return dist

def cv_emd(X, Y):
    return CalcEMD2(X, Y, CV_DIST_L2)

def safe_pc_diff(true, other):
    if true == 0:
        if other == 0:
            return 0.0
        else:
            return 100.0*other/abs(other)
    else:
        return 100.0*((other - true) / true)

def test_emd(B):
    n = 0
    t_tp = 0.0
    t_lp = 0.0
    t_cv = 0.0
    max_diff_lp = 0
    max_diff_cv = 0
    t_diff_lp = 0
    t_diff_cv = 0
    for X in B:
        for Y in B:
            n += 1
            start = time.time()
            d_tp = tp_emd(X, Y)
            t_tp += (time.time() - start)

            start = time.time()
            d_lp = lp_emd(X, Y)
            t_lp += (time.time() - start)

            start = time.time()
            X32 = np.asarray(np.hstack([np.ones((len(X), 1))/float(len(X)), X]), dtype=np.float32)
            Y32 = np.asarray(np.hstack([np.ones((len(Y), 1))/float(len(Y)), Y]), dtype=np.float32)
            d_cv = cv_emd(X32, Y32)
            t_cv += (time.time() - start)

            D = cdist(X, Y, 'euclidean')
            D32 = cdist(X32.astype(float), Y32.astype(float), 'euclidean')
            print np.average(D - D32)

            diff_lp = safe_pc_diff(d_tp, d_lp)
            diff_cv = safe_pc_diff(d_tp, d_cv)
            if abs(diff_lp) > abs(max_diff_lp):
                max_diff_lp = diff_lp
            if abs(diff_cv) > abs(max_diff_cv):
                max_diff_cv = diff_cv
            t_diff_lp += diff_lp
            t_diff_cv += diff_cv
            print
            print 'Avg. TP Time: %f' % (t_tp / n)
            print 'Avg. LP Time: %f' % (t_lp / n)
            print 'Avg. CV Time: %f' % (t_cv / n)
            print 'Max LP Diff: %.2f%%' % max_diff_lp
            print 'Max CV Diff: %.2f%%' % max_diff_cv
            print 'Avg. LP Diff: %.2f%%' % (t_diff_lp / n)
            print 'Avg. CV Diff: %.2f%%' % (t_diff_cv / n)

def main():
    data = get_dataset('trx')
    #test_nsk(data.bags)
    K1 = test_emd(data.bags)
    #B = [np.array([[1, 0], [1, 1], [0, 1], [0, 0]])]
    #test_nsk(B)

if __name__ == '__main__':
    main()
