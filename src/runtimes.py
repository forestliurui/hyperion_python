#!/usr/bin/env python
import os
import yaml
import numpy as np
from collections import defaultdict

from resampling import NullResamplingConfiguration
from server import load_config, UnfinishedException
from folds import FoldConfiguration
from data import get_dataset

def _compute_basic_time(tasks, fold_config, experiment_name,
        experiment_id, parameter_id, parameter_sets):

    outer_times = []
    for f, (outer_train, outer_test) in fold_config.get_next_level():
        for pset in parameter_sets:
            config = NullResamplingConfiguration(outer_train)
            keys = [(experiment_name, experiment_id, resampled,
                     outer_test, parameter_id, pset)
                    for resampled in config.get_all_resampled()]
            times = [tasks[key].get_statistic('training_time') for key in keys]
            outer_times.append(np.sum(times))

    return np.average(outer_times)

def compute_basic_runtime(configuration, tasks, parameters, outputfile):

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
            if any(l.startswith(line) for l in existing_lines):
                print 'Skipping existing line "%s"' % line
                continue

            try:
                stat = _compute_basic_time(tasks, fold_config,
                    experiment_name, experiment_id,
                    parameter_id, parameter_sets)
            except UnfinishedException:
                print 'Skipping unfinished line "%s"' % line
                continue

            line += (',%f\n' % stat)
            print line,
            with open(outputfile, 'a+') as f:
                f.write(line)

EXPERIMENTS = {
    'mi_kernels' : compute_basic_runtime,
}

def compute_runtimes(configuration_file, results_directory, outputfile):
    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    experiment_name = configuration['experiment_name']
    if experiment_name not in EXPERIMENTS:
        raise ValueError('No support for "%s"' % experiment_name)

    stat_func = lambda *args: EXPERIMENTS[experiment_name](*args)

    tasks, parameters = load_config(configuration_file, results_directory)

    stat_func(configuration, tasks, parameters, outputfile)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    compute_runtimes(*args, **options)
