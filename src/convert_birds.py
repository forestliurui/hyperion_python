#!/usr/bin/env python
"""
Converts the bird dataset to C4.5 format
(creates a one-vs-rest dataset for each bird)
see http://web.engr.oregonstate.edu/~briggsf/kdd2012datasets/hja_birdsong/
"""

IDS = {
    'BRCR':  1,
    'WIWR':  2,
    'PSFL':  3,
    'RBNU':  4,
    'DEJU':  5,
    'OSFL':  6,
    'HETH':  7,
    'CBCH':  8,
    'VATH':  9,
    'HEWA': 10,
    'SWTH': 11,
    'HAFL': 12,
    'WETA': 13,
}

FEATURES = 'hja_birdsong_features.txt'
LABELS = 'hja_birdsong_instance_labels.txt'

def main():
    for bird in IDS.keys():
        print 'Converting %s...' % bird
        convert_bird(bird)
    print 'Done!'

def convert_bird(bird):
    bird_id = IDS[bird]
    iids = set()
    bids = set()
    datalines = []
    nfeatures = None
    with open(FEATURES, 'r') as ffile:
        with open(LABELS, 'r') as lfile:
            for iid, (fline, lline) in enumerate(zip(ffile, lfile)):
                if iid == 0: continue # skip first line
                iids.add(iid)

                # Get bag id
                fparts = fline.strip().split(',')
                bid = int(fparts[0])
                bids.add(bid)
                fv = fparts[1:]
                if nfeatures is None:
                    nfeatures = len(fv)
                else:
                    assert (len(fv) == nfeatures)

                # Get instance label
                _, blabel = lline.strip().split(',')
                ilabel = int(int(blabel) == bird_id)

                newline = ','.join(map(str, [bid, iid]) + fv + [str(ilabel)])
                datalines.append(newline + '.')

    with open('%s.data' % bird, 'w+') as datafile:
        datafile.write('\n'.join(datalines))

    with open('%s.names' % bird, 'w+') as namesfile:
        namesfile.write('0,1.\n')
        bag_ids = 'bag_id: %s.\n' % (','.join(map(str, bids)))
        namesfile.write(bag_ids)
        instance_ids = 'instance_id: %s.\n' % (','.join(map(str, iids)))
        namesfile.write(instance_ids)
        for i in range(nfeatures):
            namesfile.write('f%d: continuous.\n' % (i+1))

if __name__ == '__main__':
    main()

