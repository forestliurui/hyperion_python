#!/usr/bin/env python

from data import get_dataset, ViewBuilder, VIEWS_PATH

def main(base_dataset, new_dataset):
    dset = get_dataset(base_dataset)
    view = ViewBuilder(new_dataset, base_dataset)

    for i, ((bid, iid), yi) in enumerate(zip(dset.instance_ids, dset.instance_labels)):
        view.add(bid, iid, 'b%d' % i, 'i%d' % i, yi)

    view.save(VIEWS_PATH[0])

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog base-dataset new-dataset")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
