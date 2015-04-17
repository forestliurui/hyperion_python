#!/usr/bin/env python
"""
Code for creating and loading folds
(folds are just special views of a dataset)
"""
import numpy as np
from random import shuffle
from itertools import izip_longest
from sklearn.cross_validation import StratifiedKFold

import data

FOLD_DIR = 'folds'
data.VIEWS_PATH.append(FOLD_DIR)

TEST_SUFFIX = '.test'
TRAIN_SUFFIX = '.train'
FOLD_FORMAT = '%s.fold_%04d_of_%04d%s'

class FoldConfiguration(object):

    def __init__(self, dataset, *fold_levels):
        self.dataset = dataset
        self.fold_levels = fold_levels
        self.params = None

    def get_settings(self):
        if self.params is None:
            params = []
            if len(self.fold_levels) > 0:
                fold_levels = list(self.fold_levels)
                folds = fold_levels.pop(0)
                if folds is None or folds <= 1:
                    folds = _get_total_folds(self.dataset)

                for f in range(folds):
                    fold_dset = fold_name(self.dataset, f, folds, train=True)
                    sub_config = FoldConfiguration(fold_dset, *fold_levels)
                    params.append({'fold': (f,)})
                    for setting in sub_config.get_settings():
                        combined = (f,) + setting['fold']
                        params.append({'fold': combined})

            self.params = params

        return self.params

    def get_train_and_test(self, fold=None):
        folds = [pair for pair in zip(fold, self.fold_levels)]
        dataset = self.dataset
        while len(folds) > 1:
            f, total = folds.pop(0)
            dataset = fold_name(dataset, f, total, train=True)

        f, total = folds.pop(0)
        train = fold_name(dataset, f, total, train=True)
        test  = fold_name(dataset, f, total, train=False)
        return train, test

    def get_all_train_and_test(self):
        return [self.get_train_and_test(**setting)
                for setting in self.get_settings()]

    def get_next_level(self, prefix=tuple()):
        if len(prefix) >= len(self.fold_levels):
            raise ValueError('There is not next level.')

        valid = [s['fold'] for s in self.get_settings()
                 if len(s['fold']) == len(prefix) + 1
                 and tuple(s['fold'][:-1]) == tuple(prefix)]

        return [(v, self.get_train_and_test(v)) for v in valid]

def _get_total_folds(dataset):
    # Assume "leave-one-out"
    dset = data.get_dataset(dataset)
    return len(dset.bag_ids)

def fold_name(dataset, number, total=None, train=None):
    if total is None or total <= 1:
        total = _get_total_folds(dataset)

    if train is None:
        suffix = ''
    elif train:
        suffix = TRAIN_SUFFIX
    else:
        suffix = TEST_SUFFIX

    return (FOLD_FORMAT % (dataset, number, total, suffix))

def make_folds(dataset, *fold_levels):
    completed_folds = _make_folds_rec(dataset, *fold_levels)
    for fold in completed_folds:
        fold.save(FOLD_DIR)

def _make_folds_rec(dataset, *fold_levels):
    if len(fold_levels) == 0:
        return []

    fold_levels = list(fold_levels)
    folds = fold_levels.pop(0)
    dset = data.get_dataset(dataset)
    labeled_bags = zip(list(dset.bag_ids), dset.bag_labels.flat)
    shuffle(labeled_bags)
    bags, labels = map(np.array, zip(*labeled_bags))

    # Check for leave-one-out
    if folds <= 1:
        folds = len(bags)

    if folds > len(bags):
        raise ValueError('%d folds requested, but only %d bags.'
                          % (folds, len(bags)))

    train_views = []
    test_views = []
    cross_validation = StratifiedKFold(labels, folds)
    for f, (train_fold, test_fold) in enumerate(cross_validation):
        train_fold_name = fold_name(dataset, f, folds, train=True)
        train_view = data.ViewBuilder(train_fold_name, dataset)
        for idx in train_fold:
            train_view.add(bags[idx])
        train_views.append(train_view)

        test_fold_name = fold_name(dataset, f, folds, train=False)
        test_view = data.ViewBuilder(test_fold_name, dataset)
        for idx in test_fold:
            test_view.add(bags[idx])
        test_views.append(test_view)

    completed = test_views
    for train in train_views:
        train.save(FOLD_DIR)
        completed += _make_folds_rec(train.name, *fold_levels)

    return completed

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog dataset folds1 [folds2 ...]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    try:
        if len(args) < 2: raise Exception()
        dataset = args.pop(0)
        foldlevels = map(int, args)
    except:
        parser.print_help()
        exit()

    make_folds(dataset, *foldlevels)
