#!/usr/bin/env python
from data import get_dataset
from progress import ProgressMonitor

DATASETS = ('BRCR', 'WIWR', 'PSFL', 'RBNU', 'DEJU', 'OSFL', 'HETH', 'CBCH',
'VATH', 'HEWA', 'SWTH', 'HAFL', 'WETA', 'elephant', 'fox', 'tiger', 'field',
'flower', 'mountain', 'apple~cokecan', 'banana~goldmedal',
'dirtyworkgloves~dirtyrunningshoe', 'wd40can~largespoon',
'checkeredscarf~dataminingbook', 'juliespot~rapbook',
'smileyfacedoll~feltflowerrug', 'stripednotebook~greenteabox',
'cardboardbox~candlewithholder', 'bluescrunge~ajaxorange',
'woodrollingpin~translucentbowl', 'fabricsoftenerbox~glazedwoodpot',
'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
'talk.politics.misc', 'talk.religion.misc', 'musk1', 'musk2', 'trx',
'text1', 'text2')

def class_ratio(dset):
    b = len(dset.bags)
    p = float(sum(dset.bag_labels))
    n = b - p

    # Special cases
    if n == 0:
        if p == 0:
            return 1.0
        else:
            return 0.0
    if p == 0:
        return 0.0

    return min(p/n, n/p)

STATISTICS = (
    ('name'       , lambda dset: dset.name),
    ('features'   , lambda dset: dset.instances.shape[1]),
    ('bags'       , lambda dset: len(dset.bags)),
    ('instances'  , lambda dset: dset.instances.shape[0]),
    ('class_ratio', class_ratio),
    ('bag_size'   , lambda dset: dset.instances.shape[0]/float(len(dset.bags))),
)

def main(outputfile):
    progress = ProgressMonitor(total=len(DATASETS), msg='Extracting statistics')
    with open(outputfile, 'w+') as f:
        stats = ','.join(stat for stat, _ in STATISTICS)
        f.write('#%s\n' % stats)
        for dataset in DATASETS:
            dset = get_dataset(dataset)
            dset.name = dataset
            stats = ','.join(map(str, (f(dset) for _, f in STATISTICS)))
            f.write('%s\n' % stats)
            progress.increment()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 1:
        parser.print_help()
        exit()
    main(*args, **options)
