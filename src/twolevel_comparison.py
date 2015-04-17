#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

NORMS = ('averaging', 'featurespace')

def parse_twolevel_results(parts, kernel):
    dataset, _, _, k, second_level, stat = parts
    if k != kernel: return None
    if second_level != 'rbf': return None
    return dataset, float(stat)

TECHNIQUES = {
  'twolevel': ('twolevel_%s.csv', parse_twolevel_results),
  'twolevel2': ('twolevel2_%s.csv', parse_twolevel_results),
}

def main(kernel, ranks_file, stats_dir, metric='acc'):
    techniques = list(TECHNIQUES.keys())
    stats = dict()
    stat_count = defaultdict(int)
    for technique, (stats_file, parser) in TECHNIQUES.items():
        stats_file = (stats_file % metric)
        with open(os.path.join(stats_dir, stats_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts, kernel)
                if results is None: continue
                dset, stat = results
                stats[technique, dset] = stat
                stat_count[dset] += 1

    good_datasets = [dset for dset in stat_count.keys()
                     if stat_count[dset] == len(techniques)]

    data = np.array([[stats[t, d] for d in good_datasets] for t in techniques])
    ranks = rankdata(-data, axis=0)
    avg_ranks = np.average(ranks, axis=1)
    with open(ranks_file, 'w+') as f:
        for t, r in zip(techniques, avg_ranks.flat):
            line = '%s,%d,%f\n' % (t, ranks.shape[1], r)
            f.write(line)
            print line,

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog kernel ranks-file stats-directory [metric=acc]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 3:
        parser.print_help()
        exit()
    main(*args, **options)
