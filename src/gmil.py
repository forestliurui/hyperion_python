#!/usr/bin/env python
import os
from itertools import cycle
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

from ranks import *
import data

SIVAL = ( 'apple~cokecan', 'banana~goldmedal',
'dirtyworkgloves~dirtyrunningshoe', 'wd40can~largespoon',
'checkeredscarf~dataminingbook', 'juliespot~rapbook',
'smileyfacedoll~feltflowerrug', 'stripednotebook~greenteabox',
'cardboardbox~candlewithholder', 'bluescrunge~ajaxorange',
'woodrollingpin~translucentbowl', 'fabricsoftenerbox~glazedwoodpot',)

def get_results(stats_dir, kernel):
    stats = defaultdict(dict)
    for technique, (stats_file, parser) in TECHNIQUES.items():
        sfile = os.path.join(stats_dir, stats_file)
        with open(sfile, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts, kernel)
                if results is None: continue
                dset, stat = results
                stats[technique][dset] = stat
    return stats

def get_var(dataset):
    variation = 0.0
    p = 0
    dset = data.get_dataset(dataset)
    for _, bag, y, inst_labels in dset.bag_dict.values():
        if y != True: continue
        p += 1
        pinsts = bag[np.array(inst_labels), :]
        variation += np.average(cdist(pinsts, pinsts, 'euclidean'))
    variation /= p
    return variation

def get_vars():
    variations = defaultdict(float)
    for sival in SIVAL:
        s1, s2 = sival.split('~')
        vpos = get_var('sival_%s' % s1)
        vneg = get_var('sival_%s' % s2)
        variations[sival] = (vpos + vneg) / 2
    return variations

def difference(stats, dset):
    nsk = stats['nsk'][dset]
    diff = stats['twolevel'][dset] - nsk
    if nsk == 0:
        pc_diff = 100
    else:
        pc_diff = 100 * (diff / nsk)
    return pc_diff

def main(kernel, stats_dir, outputfile=None):
    stats = get_results(stats_dir, kernel)
    variations = get_vars()

    x = np.array([variations[s] for s in SIVAL])
    y = np.array([difference(stats, s) for s in SIVAL])
    r = pearsonr(x, y)[0]
    p = pearsonr(x, y)[1]
    print 'Correlation: %f' % r
    print 'P-value: %f' % p

    X = np.column_stack([x, np.ones(x.size)])
    bestfit = np.linalg.lstsq(X, y)[0]
    xx = np.linspace(np.min(x), np.max(x), 3)
    XX = np.column_stack([xx, np.ones(xx.size)])
    yy = np.dot(XX, bestfit)

    fig = pl.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=25, edgecolor='none', color='k')
    ax.plot(xx, yy, '-', lw=3, color='k')
    
    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog kernel stats-directory [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2 or len(args) > 3:
        parser.print_help()
        exit()
    main(*args, **options)
