#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

NORMS = ('averaging', 'featurespace')

def emp_parser(kernel_type, transduction):
    def parse_emp_results(parts):
        dataset, _, k, ktype, eps, delta, seed, p, trans, stat = parts
        if k != 'emp': return None
        if ktype != kernel_type: return None
        if int(seed) != 0: return None
        if int(trans) != transduction: return None
        return dataset, float(stat)
    return parse_emp_results

def box_parser(kernel_type):
    def parse_box_results(parts):
        dataset, ktype, eps, delta, seed, stat = parts
        if ktype != kernel_type: return None
        if int(seed) != 0: return None
        return dataset, float(stat)
    return parse_box_results

TECHNIQUES = {
  'andor-emp': ('empbox_acc.csv', emp_parser('andor', 0)),
  'andor-emp-trans': ('empbox_acc.csv', emp_parser('andor', 1)),
  'and-emp': ('empbox_acc.csv', emp_parser('and', 0)),
  'and-emp-trans': ('empbox_acc.csv', emp_parser('and', 1)),
  'and': ('hardmarginbox_acc.csv', box_parser('and')),
  'andor': ('hardmarginbox_acc.csv', box_parser('andor')),
}

def main(ranks_file, stats_dir):
    techniques = list(TECHNIQUES.keys())
    stats = dict()
    stat_count = defaultdict(int)
    for technique, (stats_file, parser) in TECHNIQUES.items():
        with open(os.path.join(stats_dir, stats_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts)
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
    parser = OptionParser(usage="Usage: %prog ranks-file stats-directory")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
