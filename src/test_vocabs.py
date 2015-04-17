#!/usr/bin/env python
"""
Script for testing kernel code before
running the experimental code.
"""
from sklearn.metrics import accuracy_score
import numpy as np
import time

from data import get_dataset
import vocabulary

def test_vocab(data, similarity):
    parameters = {
        'regularization' : 'L2',
        'vocab_similarity' : similarity,
        'vocab_type' : 'instances',
        'vocab_gamma' : 1e-3,
        'kernel' : 'linear',
        'C' : 1e6
    }
    eSVM = vocabulary.EmbeddedSpaceSVM(**parameters)
    eSVM.fit(data.bags, data.bag_labels)
    preds = eSVM.predict()
    print accuracy_score(data.bag_labels, (preds > 0))

def main():
    data = get_dataset('musk1')
    test_vocab(data, 'miles')
    test_vocab(data, 'yards')

if __name__ == '__main__':
    main()
