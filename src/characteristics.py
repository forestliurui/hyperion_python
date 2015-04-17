#!/usr/bin/env python
import os
from itertools import cycle
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr

import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

from ranks import *

COLORS = ('darkblue', 'darkred', 'darkgreen', 'darkgoldenrod', 'indigo')

DSET_STATS = 'dataset_stats.csv'
COLS = 3

CHAR_MAP = {
    'features'   : ('Features', (-5, 250)),
    'bags'       : ('Bags', (0, 600)),
    'instances'  : ('Instances', (-100, 30000)),
    'class_ratio': ('Class Ratio', (0.05, 1.05)),
    'bag_size'   : ('Avg. Bag Size', (-10, 160)),
}

def get_dataset_characteristics(stats_dir):
    stats = defaultdict(dict)
    sfile = os.path.join(stats_dir, DSET_STATS)
    with open(sfile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                parts = line[1:].strip().split(',')
                if parts[0] != 'name':
                    raise ValueError('Expected first characteristic to be dataset name')
                chars = parts[1:]
            else:
                if chars is None:
                    raise Exception('No characteristic names found before statistics')
                parts = line.strip().split(',')
                name = parts[0]
                for char, val in zip(chars, map(float, parts[1:])):
                    stats[char][name] = val
    return stats

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

def main(kernel, stats_dir, outputfile=None):
    characteristics = get_dataset_characteristics(stats_dir)
    stats = get_results(stats_dir, kernel)

    fig = pl.figure(figsize=(16,8))
    n = len(characteristics)
    rows = int(np.ceil(float(n)/COLS))
    for i, (char, cvals) in enumerate(characteristics.items(), 1):
        ax = fig.add_subplot(rows, COLS, i)
        ax.set_xlabel(CHAR_MAP.get(char, (char,))[0])
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.5, 1.05)
        xlims = CHAR_MAP.get(char, (None, None))[1]
        if xlims is not None:
            ax.set_xlim(*xlims)

        for col, (tech, tvals) in zip(cycle(COLORS), stats.items()):
            intersection = sorted(set(cvals.keys()) & set(tvals.keys()))
            x = np.array([cvals[d] for d in intersection])
            y = np.array([tvals[d] for d in intersection])
            #r = pearsonr(x, y)[0]
            X = np.column_stack([x, np.ones(x.size)])
            bestfit = np.linalg.lstsq(X, y)[0]
            xx = np.linspace(np.min(x), np.max(x), 3)
            XX = np.column_stack([xx, np.ones(xx.size)])
            yy = np.dot(XX, bestfit)
            ax.scatter(x, y, s=15, edgecolor='none', color=col)
            ax.plot(xx, yy, '-', lw=3, color=col, label=tech)
        ax.legend(loc='lower right', ncol=2)

    pl.subplots_adjust(hspace=0.25)

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
