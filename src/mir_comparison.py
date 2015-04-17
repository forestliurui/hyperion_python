#!/usr/bin/env python
import os
import numpy as np
from scipy.stats import wilcoxon
from collections import defaultdict

def parse_nsk_results(parts, kernel):
    tech, dataset, k, stat = parts
    if tech != 'mirk': return None
    if (kernel + '_av') != k: return None
    return dataset, float(stat)

def parse_twolevel_results(parts, kernel):
    dataset, _, _, k, second_level, stat = parts
    if k != kernel: return None
    if second_level != 'rbf': return None
    return dataset, float(stat)

TECHNIQUES = {
    'nsk' : ('mir_other_r2.csv', parse_nsk_results),
    'twolevel' : ('mir_twolevel_r2.csv', parse_twolevel_results),
}
TS = sorted(TECHNIQUES.keys())

def main(stat_dir, kernel):
    stats = defaultdict(dict)
    for t, (tfile, parser) in TECHNIQUES.items():
        stat_file = os.path.join(stat_dir, tfile)
        with open(stat_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts, kernel)
                if results is None: continue
                dset, stat = results
                stats[dset][t] = stat

    S = np.array([[stats[dset][t] for t in TS]
                  for dset in stats.keys()
                    if all((t in stats[dset])
                           for t in TS)])

    wins = (S[:, 0] > S[:, 1]) + 0.5*(S[:, 0] == S[:, 1])
    winrate = np.average(wins)
    if winrate > 0.5:
        symbol = '>'
    elif winrate < 0.5:
        symbol = '<'
    else:
        symbol = '='

    print np.column_stack([S, wins])
    print '%s %s %s' % (TS[0], symbol, TS[1])
    print 'Winrate for %s: %f' % (TS[0], winrate)
    print 'Wilcoxon p-value: %f' % wilcoxon(S[:, 0], S[:, 1])[1]

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog stat-dir kernel")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
