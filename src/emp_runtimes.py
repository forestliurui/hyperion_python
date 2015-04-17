#!/usr/bin/env python
import os
import yaml
import numpy as np
from collections import defaultdict

FOLDS = 10
# The amount of kernel computation time
# corresponding to one outer training fold
FACTOR = (float(FOLDS - 1)/FOLDS)**2

BIRDS = ('BRCR', 'CBCH', 'DEJU', 'HAFL', 'HETH', 'HEWA', 'OSFL',
         'PSFL', 'RBNU', 'SWTH', 'VATH', 'WETA', 'WIWR',)

def compute_runtimes(emptimes, hardmargintimes, outputfile):
    if os.path.exists(outputfile):
        with open(outputfile, 'r+') as f:
            existing_lines = [line.strip() for line in f]
    else:
        existing_lines = []

    ktimes = dict()
    with open(hardmargintimes, 'r+') as f:
        for line in f:
            dset, ktype, eps, delta, seed, time = line.strip().split(',')
            eps, delta, time = map(float, (eps, delta, time))
            seed = int(seed)
            ktimes[dset, ktype, eps, delta, seed] = FACTOR*time
            if dset in BIRDS:
                for d in BIRDS:
                    ktimes[d, ktype, eps, delta, seed] = FACTOR*time

    with open(emptimes, 'r+') as f:
        for line in f:
            dset, alg, k, ktype, eps, delta, seed, p, trans, time = line.strip().split(',')
            eps, delta, time = map(float, (eps, delta, time))
            seed, p, trans = map(int, (seed, p, trans))

            line = ('%s,%s,%s,%s,%f,%f,%d,%d,%d' %
                (dset, alg, k, ktype, eps, delta, seed, p, trans))
            if any(l.startswith(line) for l in existing_lines):
                continue

            time += ktimes[dset, ktype, eps, delta, seed]
            line += (',%f\n' % time)
            print line,
            with open(outputfile, 'a+') as f:
                f.write(line)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog emptimes hardmargintimes outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    compute_runtimes(*args, **options)
