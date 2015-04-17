#!/usr/bin/env python
import numpy as np
from scipy.stats import wilcoxon
from collections import defaultdict

NORMS = ('averaging', 'featurespace')

def parse_nsk_results(parts, kernel):
    dataset, _, _, k, normalization, stat = parts
    if k != kernel: return None
    return dataset, normalization, float(stat)

def main(stat_file, kernel):
    stats = defaultdict(dict)
    with open(stat_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            results = parse_nsk_results(parts, kernel)
            if results is None: continue
            dset, norm, stat = results
            stats[dset][norm] = stat

    S = np.array([[stats[dset][norm] for norm in NORMS]
                  for dset in stats.keys()
                    if all((norm in stats[dset])
                           for norm in NORMS)])

    wins = (S[:, 0] > S[:, 1]) + 0.5*(S[:, 0] == S[:, 1])
    winrate = np.average(wins)
    if winrate > 0.5:
        symbol = '>'
    elif winrate < 0.5:
        symbol = '<'
    else:
        symbol = '='

    print np.column_stack([S, wins])
    print '%s %s %s' % (NORMS[0], symbol, NORMS[1])
    print 'Winrate for %s: %f' % (NORMS[0], winrate)
    print 'Wilcoxon p-value: %f' % wilcoxon(S[:, 0], S[:, 1])[1]

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog stat-file kernel")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
