#!/usr/bin/env python
import os
import glob

import data
import folds

PREFIX = 'mir_'

# Code to load old MIR-style folds
def get_outer_folds(folddir, dataset):
    regex = os.path.join(folddir, '%s*.ofold' % dataset)
    return glob.glob(regex)

def get_inner_folds(folddir, outerfold, dataset):
    regex = os.path.join(folddir, '%s_%04d*.ifold' % (dataset, outerfold))
    return glob.glob(regex)

def get_outer_fold(folddir, dataset, outerfold):
    path = os.path.join(folddir, '%s_%04d.ofold' % (dataset, outerfold))
    with open(path, 'r') as f:
        return map(int, f)

def get_inner_fold(folddir, dataset, outerfold, innerfold):
    path = os.path.join(folddir, '%s_%04d_%04d.ifold' % (dataset, outerfold, innerfold))
    with open(path, 'r') as f:
        return map(int, f)

def convert_folds(dataset, folddir):
    pdataset = PREFIX + dataset
    outer_folds = len(get_outer_folds(folddir, dataset))
    outer_keys = set()
    for ofold in range(outer_folds):
        outer_keys |= set(get_outer_fold(folddir, dataset, ofold))

    # Create meta-view file
    view = data.ViewBuilder(dataset, pdataset)
    for key in outer_keys: view.add(str(key))
    view.save(data.VIEWS_PATH[0])

    for ofold in range(outer_folds):
        otrain = folds.fold_name(dataset, ofold, outer_folds, True)
        otest  = folds.fold_name(dataset, ofold, outer_folds, False)

        # Outer View Files
        otrain_view = data.ViewBuilder(otrain, dataset)
        otest_view  = data.ViewBuilder(otest, dataset)
        outer_test = set(get_outer_fold(folddir, dataset, ofold))
        outer_train = outer_keys - outer_test
        for okey in outer_test: otest_view.add(str(okey))
        for okey in outer_train: otrain_view.add(str(okey))
        otest_view.save(folds.FOLD_DIR)
        otrain_view.save(folds.FOLD_DIR)

        inner_folds = len(get_inner_folds(folddir, ofold, dataset))
        for ifold in range(inner_folds):
            itrain = folds.fold_name(otrain, ifold, inner_folds, True)
            itest  = folds.fold_name(otrain, ifold, inner_folds, False)

            # Inner View Files
            itrain_view = data.ViewBuilder(itrain, otrain)
            itest_view = data.ViewBuilder(itest, otrain)
            inner_test = set(get_inner_fold(folddir, dataset, ofold, ifold))
            inner_train = outer_train - inner_test
            for ikey in inner_test: itest_view.add(str(ikey))
            for ikey in inner_train: itrain_view.add(str(ikey))
            itest_view.save(folds.FOLD_DIR)
            itrain_view.save(folds.FOLD_DIR)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog dataset mir-fold-dir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2:
        parser.print_help()
        exit()
    convert_folds(*args)
