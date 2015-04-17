#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

from ranks import *

TECHNIQUES = {
  'nsk' : ('nsk_times.csv', parse_nsk_results),
  'miGraph' : ('migraph_times.csv', parse_miGraph_results),
  #'MIGraph' : ('capital_MIgraph_times.csv', parse_MIGraph_results),
  'twolevel': ('twolevel2_times.csv', parse_twolevel_results),
  'emd': ('emd_times.csv', parse_emd_results),
  #'kemd': ('kemd_times.csv', parse_kemd_results),
  'yards': ('yards_times.csv', parse_yards_results),
  'box': ('empbox_total_times.csv', parse_box_results),
  #'miles': ('miles_times.csv', parse_miles_results),
  #'og_yards': ('og_yards_times.csv', lambda p, k: parse_yards_results(p, 'linear')),
  #'og_miles': ('og_miles_times.csv', lambda p, k: parse_miles_results(p, 'linear')),
}

def main(kernel, ranks_file, stats_dir):
    techniques = list(TECHNIQUES.keys())
    stats = dict()
    stat_count = defaultdict(int)
    for technique, (stats_file, parser) in TECHNIQUES.items():
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
    ranks = rankdata(data, axis=0)
    avg_ranks = np.average(ranks, axis=1)
    with open(ranks_file, 'w+') as f:
        for t, r in zip(techniques, avg_ranks.flat):
            line = '%s,%d,%f\n' % (t, ranks.shape[1], r)
            f.write(line)
            print line,

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog kernel ranks-file stats-directory")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    main(*args, **options)
