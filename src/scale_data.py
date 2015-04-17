#!/usr/bin/env python
"""
Scales data for the box kernel
"""
import os
import numpy as np

from data import get_dataset

NAMES_FMT = '%s.names'
DATA_FMT = '%s.data'

def main(dataset, factor, outputdir):
    factor = float(factor)
    dset = get_dataset(dataset)
    X = (factor*dset.instances).astype(int)

    # Remove irrelevant columns (all feature values identical)
    relevant = np.nonzero(np.max(X, axis=0) - np.min(X, axis=0))[0]
    X = X[:, relevant]

    namesfile = os.path.join(outputdir, NAMES_FMT % dataset)
    datafile = os.path.join(outputdir, DATA_FMT % dataset)

    with open(namesfile, 'w+') as f:
        f.write('0,1.\n')
        f.write('bag_id: %s.\n' % ','.join(dset.bag_ids))
        f.write('instance_id: %s.\n'
            % ','.join([iid[1] for iid in dset.instance_ids]))
        for i in range(X.shape[1]):
            f.write('f%d: continuous.\n' % (i+1))

    with open(datafile, 'w+') as f:
        for (bid, iid), xx, y in zip(dset.instance_ids, X, dset.instance_labels):
            xs = ','.join(map(str, xx))
            f.write('%s,%s,%s,%d.\n' % (bid, iid, xs, y))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog dataset factor outputdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    main(*args, **options)
