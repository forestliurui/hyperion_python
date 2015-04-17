#!/usr/bin/env python
import os
import yaml
import numpy as np
from collections import defaultdict
try:
    from sklearn.metrics import roc_auc_score as auc_score
except:
    from sklearn.metrics import auc_score
from sklearn.metrics import r2_score

from resampling import NullResamplingConfiguration
from server import load_config, UnfinishedException
from folds import FoldConfiguration
from data import get_dataset

def accuracy(ytrue, ypred):
    wrong = np.logical_xor((ytrue > 0), (ypred > 0)).astype(int)
    return 1.0 - (float(np.sum(wrong)) / len(wrong))

def balanced_accuracy(ytrue, ypred):

    gold_pos = (ytrue > 0)
    true_pos = np.logical_and(gold_pos, (ypred > 0)).astype(int)
    sensitivity = float(np.sum(true_pos)) / np.sum(gold_pos)

    gold_neg = (ytrue <= 0)
    true_neg = np.logical_and(gold_neg, (ypred <= 0)).astype(int)
    specificity = float(np.sum(true_neg)) / np.sum(gold_neg)

    return (sensitivity + specificity) / 2.0

def dict_statistic_wrapper(statistic, level='bags', dataset='test'):
    def dict_statistic(true, pred):
        keys = sorted(pred.keys())
        true_array = np.array([true[k] for k in keys])
        pred_array = np.array([pred[k] for k in keys])
        return statistic(true_array, pred_array)
    dict_statistic.level = level
    dict_statistic.dataset = dataset
    return dict_statistic

def _true_labels(dataset, level='bags'):
    dset = get_dataset(dataset)
    if level.startswith('b'):
        return dict(zip(dset.bag_ids, dset.bag_labels.flat))
    elif level.startswith('i'):
        return dict(zip(dset.instance_ids, dset.instance_labels.flat))
    else:
        raise ValueError('Bad level type "%s"' % level)

def dictavg(dicts):
    collector = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if not isinstance(v, (int, long, float, complex)):
                continue
            collector[k].append(v)
    return dict([(k, np.average(v))
        for k, v in collector.items()])

def dictzip(*dicts):
    full = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            full[k].append(v)
    return full

def _ensemble_prediction(tasks, bagging_config, experiment_name, experiment_id,
        parameter_id, parameter_set, test_dataset, level, train_or_test='test'):
    keys = [(experiment_name, experiment_id,
             resampled, test_dataset, parameter_id, parameter_set)
            for resampled in bagging_config.get_all_resampled()]
    all_preds = dictzip(*[tasks[key].get_predictions(level, train_or_test)
                          for key in keys])
    ensemble_preds = {}
    for k, preds in all_preds.items():
        ensemble_preds[k] = np.average(preds)
    return ensemble_preds

def _compute_basic_best(tasks, fold_config, experiment_name,
        experiment_id, parameter_id, parameter_sets, statistic):

    stats = {}
    for pset in parameter_sets:
        outer_predictions = {}
        outer_labels = {}
        for f, (outer_train, outer_test) in fold_config.get_next_level():
            config = NullResamplingConfiguration(outer_train)
            preds = _ensemble_prediction(tasks, config,
                                         experiment_name, experiment_id,
                                         parameter_id, pset, outer_test, 'bags')
            outer_predictions.update(preds)
            outer_labels.update(_true_labels(outer_test, 'bags'))
        stat = statistic(outer_labels, outer_predictions)
        stats[pset] = stat

    return stats

def compute_basic_statistic(configuration, tasks, parameters, outputfile, statistic):

    if os.path.exists(outputfile):
        with open(outputfile, 'r+') as f:
            existing_lines = [line.strip() for line in f]
    else:
        existing_lines = []

    experiment_name = configuration['experiment_name']
    experiment_key = configuration['experiment_key']

    for experiment in configuration['experiments']:
        experiment_id = tuple(experiment[k] for k in experiment_key)

        def _resolve(field_name):
            return experiment.get(field_name, configuration[field_name])

        experiment_format = _resolve('experiment_key_format')

        param_config = parameters[experiment_id]
        parameter_ids = [s['parameter_id']
                         for s in param_config.get_settings()]
        parameter_format = _resolve('parameter_key_format')

        dataset = experiment['dataset']
        folds = _resolve('folds')
        fold_config = FoldConfiguration(dataset, *folds)

        combined_format = experiment_format + parameter_format
        line_format = ','.join(combined_format)

        for parameter_id, parameter_sets in param_config.get_parameter_sets():
            line = (line_format % (experiment_id + parameter_id ))

            try:
                stats = _compute_basic_best(tasks, fold_config,
                    experiment_name, experiment_id,
                    parameter_id, parameter_sets, statistic)
            except UnfinishedException:
                print 'Skipping unfinished line "%s"' % line
                continue

            for s, stat in stats.items():
                params = param_config.get_parameters(parameter_id, s)
                pstr = '|'.join(['%s:%f' % (k, v) for k, v in sorted(params.items())
                                 if isinstance(v, (int, long, float, complex))])

                line = (line_format % (experiment_id + parameter_id ))
                line += (',%d,' % s)
                if any(l.startswith(line) for l in existing_lines):
                    print 'Skipping existing line "%s..."' % line
                    continue

                line += ('%s,%f\n' % (pstr, stat))
                print line,
                with open(outputfile, 'a+') as f:
                    f.write(line)


EXPERIMENTS = {
    'mi_kernels' : compute_basic_statistic,
}

STATISTICS = {
    'accuracy' : accuracy,
    'auc' : auc_score,
    'balanced_accuracy' : balanced_accuracy,
    'r2' : r2_score,
}

def compute_statistics(configuration_file, results_directory, train_or_test, statistic, outputfile):
    bag_or_instance = 'bag'
    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    experiment_name = configuration['experiment_name']
    if experiment_name not in EXPERIMENTS:
        raise ValueError('No statistics supported for "%s"' % experiment_name)
    if not (bag_or_instance.startswith('b') or bag_or_instance.startswith('i')):
        raise ValueError('Third argument must be "bag" or "instance"')
    if train_or_test not in ('train', 'test'):
        raise ValueError('Third argument must be "train" or "test"')
    if statistic not in STATISTICS:
        raise ValueError('Statistic must be one of: "%s"' % ', '.join(STATISTICS.keys()))

    stat = dict_statistic_wrapper(STATISTICS[statistic], bag_or_instance, train_or_test)
    stat_func = lambda *args: EXPERIMENTS[experiment_name](*args, statistic=stat)

    tasks, parameters = load_config(configuration_file, results_directory)

    stat_func(configuration, tasks, parameters, outputfile)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir train_or_test statistic outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 5:
        parser.print_help()
        exit()
    compute_statistics(*args, **options)
