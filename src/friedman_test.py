#!/usr/bin/env python
import numpy as np
from scipy.stats import chi2, f as fdist

import nemenyi

def main(rank_file):
    ranks = []
    N = None
    with open(rank_file, 'r') as f:
        for line in f:
            _, dsets, r = line.strip().split(',')
            if N is None:
                N = int(dsets)
            ranks.append(float(r))

    k = len(ranks)
    ranksum = np.sum(np.square(ranks))
    friedman_statistic = (12.0*N/(k*(k+1)))*(ranksum - ((k * (k+1)**2) / 4.0))
    f_value = ((N - 1)*friedman_statistic) / (N*(k-1) - friedman_statistic)
    print 'p-value (Friedman Statistics): %f' % (1.0 - chi2.cdf(friedman_statistic, k-1))
    print '     p-value (Iman/Davenport): %f' % (1.0 - fdist.cdf(f_value, k-1, (k-1)*(N-1)))
    for alpha in (0.10, 0.05, 0.01):
        print 'CD_%.2f: %f' % (alpha, nemenyi.critical_difference(alpha, k, N))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog rank-file")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 1:
        parser.print_help()
        exit()
    main(*args, **options)
