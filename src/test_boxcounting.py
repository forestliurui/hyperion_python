#!/usr/bin/env python
"""
Script for testing kernel code before
running the experimental code.
"""
import numpy as np
from data import get_dataset
from scipy.io import savemat

from boxkernel import k as boxk

def test_k_and(bags):
    K = boxk(bags, bags, 0.01, 0.1, 0.01, bags)
    savemat('boxkand.mat', {'K' : K})

def main():
    data = get_dataset('trx')
    test_k_and(data.bags)

if __name__ == '__main__':
    main()
