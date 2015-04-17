#!/usr/bin/env python
import os
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats.mstats import rankdata

def parse_nsk_results(parts, kernel):
    dataset, _, _, k, normalization, stat = parts
    if k != kernel: return None
    if normalization != 'averaging': return None
    return dataset, float(stat)

def parse_miGraph_results(parts, kernel):
    dataset, _, _, k, delta, stat = parts
    if k != kernel: return None
    if float(delta) != 0: return None
    return dataset, float(stat)

def parse_twolevel_results(parts, kernel):
    dataset, _, _, k, second_level, stat = parts
    if k != kernel: return None
    if second_level != 'rbf': return None
    return dataset, float(stat)

def parse_emd_results(parts, kernel):
    dataset, _, k, _, stat = parts
    if k != ('distance_%s' % kernel): return None
    return dataset, float(stat)

def parse_yards_results(parts, kernel):
    dataset, _, _, _, _, k, stat = parts
    if k != kernel: return None
    return dataset, float(stat)

def parse_box_results(parts, kernel):
    dataset, _, k, ktype, eps, delta, seed, p, trans, stat = parts
    if k != 'emp': return None
    if ktype != 'andor': return None
    if int(seed) != 0: return None
    if int(trans) != 0: return None
    return dataset, float(stat)

TECHNIQUES = (
  ('emd',      'emd_%s.csv',      parse_emd_results),
  ('twolevel', 'twolevel_%s.csv', parse_twolevel_results),
  ('nsk',      'nsk_%s.csv',      parse_nsk_results),
  ('miGraph',  'migraph_%s.csv',  parse_miGraph_results),
  ('yards',    'yards_%s.csv',    parse_yards_results),
  ('box',      'empbox_%s.csv',   parse_box_results),
)

DSET_MAP = (
  ('musk1'                             , 'musk1'),
  ('musk2'                             , 'musk2'),
  ('elephant'                          , 'elephant'),
  ('fox'                               , 'fox'),
  ('tiger'                             , 'tiger'),
  ('field'                             , 'field'),
  ('flower'                            , 'flower'),
  ('mountain'                          , 'mountain'),
  ('apple~cokecan'                     , 'SIVAL01'),
  ('banana~goldmedal'                  , 'SIVAL02'),
  ('bluescrunge~ajaxorange'            , 'SIVAL03'),
  ('cardboardbox~candlewithholder'     , 'SIVAL04'),
  ('checkeredscarf~dataminingbook'     , 'SIVAL05'),
  ('dirtyworkgloves~dirtyrunningshoe'  , 'SIVAL06'),
  ('fabricsoftenerbox~glazedwoodpot'   , 'SIVAL07'),
  ('juliespot~rapbook'                 , 'SIVAL08'),
  ('smileyfacedoll~feltflowerrug'      , 'SIVAL09'),
  ('stripednotebook~greenteabox'       , 'SIVAL10'),
  ('wd40can~largespoon'                , 'SIVAL11'),
  ('woodrollingpin~translucentbowl'    , 'SIVAL12'),
  ('alt.atheism'                       , 'Newsgroups01'),
  ('comp.graphics'                     , 'Newsgroups02'),
  ('comp.os.ms-windows.misc'           , 'Newsgroups03'),
  ('comp.sys.ibm.pc.hardware'          , 'Newsgroups04'),
  ('comp.sys.mac.hardware'             , 'Newsgroups05'),
  ('comp.windows.x'                    , 'Newsgroups06'),
  ('misc.forsale'                      , 'Newsgroups07'),
  ('rec.autos'                         , 'Newsgroups08'),
  ('rec.motorcycles'                   , 'Newsgroups09'),
  ('rec.sport.baseball'                , 'Newsgroups10'),
  ('rec.sport.hockey'                  , 'Newsgroups11'),
  ('sci.crypt'                         , 'Newsgroups12'),
  ('sci.electronics'                   , 'Newsgroups13'),
  ('sci.med'                           , 'Newsgroups14'),
  ('sci.space'                         , 'Newsgroups15'),
  ('soc.religion.christian'            , 'Newsgroups16'),
  ('talk.politics.guns'                , 'Newsgroups17'),
  ('talk.politics.mideast'             , 'Newsgroups18'),
  ('talk.politics.misc'                , 'Newsgroups19'),
  ('talk.religion.misc'                , 'Newsgroups20'),
  ('text1'                             , 'OHSUMED1'),
  ('text2'                             , 'OHSUMED2'),
  ('BRCR'                              , 'Birdsong01'),
  ('CBCH'                              , 'Birdsong02'),
  ('DEJU'                              , 'Birdsong03'),
  ('HAFL'                              , 'Birdsong04'),
  ('HETH'                              , 'Birdsong05'),
  ('HEWA'                              , 'Birdsong06'),
  ('OSFL'                              , 'Birdsong07'),
  ('PSFL'                              , 'Birdsong08'),
  ('RBNU'                              , 'Birdsong09'),
  ('SWTH'                              , 'Birdsong10'),
  ('VATH'                              , 'Birdsong11'),
  ('WETA'                              , 'Birdsong12'),
  ('WIWR'                              , 'Birdsong13'),
  ('trx'                               , 'TRX'),
)

def main(kernel, stats_dir, metric='acc'):
    statdict = defaultdict(dict)
    stat_count = defaultdict(int)
    for technique, stats_file, parser in TECHNIQUES:
        stats_file = (stats_file % metric)
        with open(os.path.join(stats_dir, stats_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts, kernel)
                if results is None: continue
                dset, stat = results
                stat = float('%.03f' % float(stat))
                statdict[dset][technique] = stat

    good_dsets = set([dset for dset, techdict in statdict.items()
                      if len(techdict) == len(TECHNIQUES)])
    maxlen = max([len(name) for d, name in DSET_MAP if d in good_dsets])
    def pad(string):
        n = len(string)
        return string + (' '*(maxlen - n + 1))

    def get_stat(d, t):
        maxval = max(statdict[d].values())
        stat = statdict[d][t]
        if maxval == stat:
            return r'& \textbf{%.03f} ' % stat
        else:
            return r'&         %.03f  ' % stat

    for d, name in DSET_MAP:
        if d not in good_dsets: continue
        values = tuple([pad(name)] + [get_stat(d, t) for t, _, _ in TECHNIQUES])
        print '    %s\\\\' % ''.join(values)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog kernel stats-directory [metric=acc]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2:
        parser.print_help()
        exit()
    main(*args, **options)
