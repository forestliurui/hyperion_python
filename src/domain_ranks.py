#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

from ranks import *

DOMAINS = {
  'audio' : ('BRCR', 'WIWR', 'PSFL', 'RBNU', 'DEJU', 'OSFL', 'HETH', 'CBCH',
             'VATH', 'HEWA', 'SWTH', 'HAFL', 'WETA',),

  'image' : ('elephant', 'fox', 'tiger', 'field', 'flower', 'mountain',
             'apple~cokecan', 'banana~goldmedal',
             'dirtyworkgloves~dirtyrunningshoe', 'wd40can~largespoon',
             'checkeredscarf~dataminingbook', 'juliespot~rapbook',
             'smileyfacedoll~feltflowerrug', 'stripednotebook~greenteabox',
             'cardboardbox~candlewithholder', 'bluescrunge~ajaxorange',
             'woodrollingpin~translucentbowl',
             'fabricsoftenerbox~glazedwoodpot',),

  'text' : ('alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
            'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
            'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
            'sci.electronics', 'sci.med', 'sci.space',
            'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc',
            'talk.religion.misc',),

  'chemistry' : ('musk1', 'musk2', 'trx'),
}

def main(domain, kernel, ranks_file, stats_dir, metric='acc'):
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
                     if stat_count[dset] == len(techniques)
                        and dset in DOMAINS[domain]]

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
    parser = OptionParser(usage="Usage: %prog domain kernel ranks-file stats-directory [metric=acc]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 4:
        parser.print_help()
        exit()
    if args[0] not in DOMAINS.keys():
        parser.print_help()
        print '"domain" must be one of: %s' % ', '.join(DOMAINS.keys())
        exit()
    main(*args, **options)
