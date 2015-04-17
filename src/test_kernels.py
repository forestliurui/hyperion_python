#!/usr/bin/env python
"""
Script for testing kernel code before
running the experimental code.
"""
import numpy as np
from data import get_dataset

import kernel

def is_pos_def(K):
    return np.all(np.linalg.eigvals(K) > 0)

def test_nsk(B):
    k = kernel.by_name('nsk', base_kernel='rbf', gamma=1e-1, normalization='averaging')
    K = k(B, B)
    print K
    print is_pos_def(K)

def main():
    data = get_dataset('musk1')
    test_nsk(data.bags)

if __name__ == '__main__':
    main()
