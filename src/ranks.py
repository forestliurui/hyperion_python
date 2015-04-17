#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

def parse_nsk_results(parts, kernel):
    dataset, _, _, k, normalization, stat = parts
    if k != kernel: return None
    if normalization != 'averaging': return None
    return dataset, float(stat)

def parse_miGraph_results(parts, kernel):
    dataset, _, _, k, delta, stat = parts
    if k != kernel: return None
    if float(delta) != 0: return None
    return dataset, float(stat)

def parse_MIGraph_results(parts, kernel):
    dataset, _, _, k, _, epsilon, stat = parts
    if k != kernel: return None
    if float(epsilon) != 0: return None
    return dataset, float(stat)

def parse_twolevel_results(parts, kernel):
    dataset, _, _, k, second_level, stat = parts
    if k != kernel: return None
    if second_level != 'rbf': return None
    return dataset, float(stat)

def parse_emd_results(parts, kernel):
    dataset, _, k, _, stat = parts
    if k != ('distance_%s' % kernel): return None
    return dataset, float(stat)

def parse_kemd_results(parts, kernel):
    dataset, _, k, _, _, stat = parts
    if k != ('distance_%s' % kernel): return None
    return dataset, float(stat)

def parse_yards_results(parts, kernel):
    dataset, _, _, _, _, k, stat = parts
    if k != kernel: return None
    return dataset, float(stat)

def parse_miles_results(parts, kernel):
    dataset, _, _, _, _, k, stat = parts
    if k != kernel: return None
    return dataset, float(stat)

def parse_box_results(parts, kernel):
    dataset, _, k, ktype, eps, delta, seed, p, trans, stat = parts
    if k != 'emp': return None
    if ktype != 'andor': return None
    if int(seed) != 0: return None
    if int(trans) != 0: return None
    return dataset, float(stat)

TECHNIQUES = {
  'nsk' : ('nsk_%s.csv', parse_nsk_results),
  'miGraph' : ('migraph_%s.csv', parse_miGraph_results),
  #'MIGraph' : ('capital_MIgraph_%s.csv', parse_MIGraph_results),
  'twolevel': ('twolevel2_%s.csv', parse_twolevel_results),
  'emd': ('emd_%s.csv', parse_emd_results),
  #'kemd': ('kemd_%s.csv', parse_kemd_results),
  'yards': ('yards_%s.csv', parse_yards_results),
  'box': ('empbox_%s.csv', parse_box_results),
  #'miles': ('miles_%s.csv', parse_miles_results),
  #'og_yards': ('og_yards_%s.csv', lambda p, k: parse_yards_results(p, 'linear')),
  #'og_miles': ('og_miles_%s.csv', lambda p, k: parse_miles_results(p, 'linear')),
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
