#!/usr/bin/env python
from data import get_dataset

def main(dataset):
    dset = get_dataset(dataset)
    i, f = dset.instances.shape
    b = len(dset.bags)
    p = sum(dset.bag_labels)
    n = b - p
    print 'Dataset.............%s' % dataset
    print 'Features............%d' % f
    print 'Instances...........%d' % i
    print 'Bags................%d' % b
    print '    Positive........%d' % p
    print '    Negative........%d' % n
    print 'Avg. Instances/Bag..%.1f' % (float(i)/b)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog dataset")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 1:
        parser.print_help()
        exit()
    main(*args, **options)
