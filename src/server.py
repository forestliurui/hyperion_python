#!/usr/bin/env python
import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import random
import cherrypy
from cherrypy import expose, HTTPError
from threading import RLock
from collections import defaultdict
from random import shuffle
import numpy as np

from folds import FoldConfiguration
from progress import ProgressMonitor
from results import get_result_manager

PORT = 2114
DEFAULT_TASK_EXPIRE = 120 # Seconds
TEMPLATE = """
<html>
<head>
  <META HTTP-EQUIV="REFRESH" CONTENT="60">
  <title>%s</title>
  <style type="text/css">
    table.status {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: black;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.status td {
        border-width: 1px;
        padding: 1px;
        border-style: solid;
        border-color: black;
        text-align: center;
    }
    table.summary {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: none;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.summary td {
        border-width: 0px;
        padding: 3px;
        border-style: none;
        border-color: black;
        text-align: center;
        width: 50px;
    }
    td.tech { width: 50px; }
    td.done {
      background-color: green;
    }
    td.pending {
      background-color: yellow;
    }
    td.failed {
      background-color: red;
    }
    td.na {
      background-color: gray;
    }
  </style>
</head>
<body>
<h1>Time Remaining: %s</h1>
%s
</body>
</html>
"""

class UnfinishedException(Exception): pass

def plaintext(f):
    f._cp_config = {'response.headers.Content-Type': 'text/plain'}
    return f

class ExperimentServer(object):

    def __init__(self, tasks, params, render,
                 task_expire=DEFAULT_TASK_EXPIRE):
        self.status_lock = RLock()
        self.tasks = tasks
        self.params = params
        self.render = render
        self.task_expire = task_expire

        self.unfinished = set(self.tasks.items())

    def clean(self):
        with self.status_lock:
            self.unfinished = filter(lambda x: (not x[1].finished),
                                     self.unfinished)
            for key, task in self.unfinished:
                if (task.in_progress and
                    task.staleness() > self.task_expire):
                    task.quit()

    @expose
    def index(self):
        with self.status_lock:
            self.clean()
            return self.render(self.tasks)

    @plaintext
    @expose
    def request(self):
        with self.status_lock:
            self.clean()
            # Select a job to perform
            unfinished = list(self.unfinished)
            shuffle(unfinished)
            candidates = sorted(unfinished, key=lambda x: x[1].priority())
            if len(candidates) == 0:
                raise HTTPError(404)
            key, task = candidates.pop(0)
            task.ping()

        (experiment_name, experiment_id,
         train, test, parameter_id, parameter_set) = key
        parameters = self.params[experiment_id].get_parameters(
            parameter_id=parameter_id, parameter_set=parameter_set)
        arguments = {'key': key, 'parameters': parameters}
        return yaml.dump(arguments, Dumper=Dumper)

    @plaintext
    @expose
    def update(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.ping()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def quit(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.quit()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def fail(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.fail()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def submit(self, key_yaml=None, sub_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
            submission = yaml.load(sub_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.store_results(submission)
                task.finish()
        return "OK"

def time_remaining_estimate(tasks, alpha=0.1):
    to_go = float(len([task for task in tasks if not task.finished]))
    finish_times = sorted([task.finish_time for task in tasks if task.finished])
    ewma = 0.0
    for interarrival in np.diff(finish_times):
        ewma = alpha*interarrival + (1.0 - alpha)*ewma

    if ewma == 0:
        return '???'

    remaining = to_go * ewma
    if remaining >= 604800:
        return '%.1f weeks' % (remaining/604800)
    elif remaining >= 86400:
        return '%.1f days' % (remaining/86400)
    elif remaining >= 3600:
        return '%.1f hours' % (remaining/3600)
    elif remaining >= 60:
        return '%.1f minutes' % (remaining/60)
    else:
        return '%.1f seconds' % remaining

def render(tasks):
    # Get dimensions
    experiment_names = set()
    experiment_ids = set()
    parameter_ids = set()
    for key in tasks.keys():
        experiment_names.add(key[0])
        experiment_ids.add(key[1])
        parameter_ids.add(key[4])

    experiment_names = sorted(experiment_names)
    experiment_title = ('Status: %s' % ', '.join(experiment_names))

    time_est = time_remaining_estimate(tasks.values())

    reindexed = defaultdict(list)
    for k, v in tasks.items():
        reindexed[k[1], k[4]].append(v)

    tasks = reindexed

    table = '<table class="status">'
    # Experiment header row
    table += '<tr><td style="border:0" rowspan="1"></td>'
    for parameter_id in parameter_ids:
        table += ('<td class="tech">%s</td>' % str(parameter_id))
    table += '</tr>\n'

    # Data rows
    for experiment_id in sorted(experiment_ids):
        table += ('<tr><td class="data">%s</td>' % str(experiment_id))
        for parameter_id in parameter_ids:
            key = (experiment_id, parameter_id)
            title = ('%s, %s' % tuple(map(str, key)))
            if key in tasks:
                table += ('<td style="padding: 0px;">%s</td>' % render_task_summary(tasks[key]))
            else:
                table += ('<td class="na" title="%s"></td>' % title)
        table += '</tr>\n'

    table += '</table>'
    return (TEMPLATE % (experiment_title, time_est, table))

def render_task_summary(tasks):
    n = float(len(tasks))
    failed = 0
    finished = 0
    in_progress = 0
    waiting = 0
    for task in tasks:
        if task.finished:
            finished += 1
        elif task.failed:
            failed += 1
        elif task.in_progress:
            in_progress += 1
        else:
            waiting += 1

    if n == finished:
        table = '<table class="summary"><tr>'
        table += ('<td class="done" title="Finished">D</td>')
        table += ('<td class="done" title="Finished">O</td>')
        table += ('<td class="done" title="Finished">N</td>')
        table += ('<td class="done" title="Finished">E</td>')
        table += '</tr></table>'
    else:
        table = '<table class="summary"><tr>'
        table += ('<td title="Waiting">%.2f%%</td>' % (100*waiting/n))
        table += ('<td class="failed" title="Failed">%.2f%%</td>' % (100*failed/n))
        table += ('<td class="pending" title="In Progress">%.2f%%</td>' % (100*in_progress/n))
        table += ('<td class="done" title="Finished">%.2f%%</td>' % (100*finished/n))
        table += '</tr></table>'
    return table

class ParameterConfiguration(object):

    def __init__(self, results_directory, experiment_name,
                 experiment_id, experiment_format,
                 parameter_key, parameter_format, parameter_configuration):
        self.results_directory = results_directory
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_format = experiment_format
        self.parameter_key = parameter_key
        self.parameter_format = parameter_format
        self.parameter_configuration = parameter_configuration

        self.param_directory = os.path.join(results_directory, experiment_name)
        if not os.path.exists(self.param_directory):
            os.mkdir(self.param_directory)

        self.settings = None
        self.param_dict = {}

    def _parameter_path(self, parameter_id):
        key = self.experiment_id
        key += parameter_id
        fmt = list(self.experiment_format)
        fmt += self.parameter_format
        format_str = '_'.join(fmt)
        filename = (format_str % key)
        filename += '.params'
        return os.path.join(self.param_directory, filename)

    def get_settings(self):
        if self.settings is None:
            self.settings = []

            for parameters in self.parameter_configuration:
                parameters = dict(**parameters)
                p_search = parameters.pop('search')
                search_type = p_search['type']
                if search_type != 'random':
                    raise ValueError('Unknown search type ""' % search_type)

                parameter_id = tuple(parameters[k] for k in self.parameter_key)
                param_path = self._parameter_path(parameter_id)

                # Load any parameters that already exist
                if os.path.exists(param_path):
                    with open(param_path, 'r') as f:
                        param_list = yaml.load(f)
                else:
                    param_list = []

                # Add additional parameter sets as needed
                for i in range(p_search['n'] - len(param_list)):
                    params = {}
                    for param, constraints in parameters.items():
                        if type(constraints) == list:
                            if (type(constraints[0]) == str
                                and constraints[0][0] == 'e'):
                                params[param] = 10**random.uniform(
                                                        *[float(c[1:])
                                                        for c in constraints])
                            else:
                                params[param] = random.uniform(*map(float, constraints))
                        else:
                            params[param] = constraints
                    param_list.append(params)

                with open(param_path, 'w+') as f:
                    f.write(yaml.dump(param_list, Dumper=Dumper))

                self.param_dict[parameter_id] = param_list
                for i in range(len(param_list)):
                    self.settings.append({'parameter_id': parameter_id,
                                          'parameter_set': i})

        return self.settings

    def get_parameters(self, parameter_id=None, parameter_set=None):
        self.get_settings() # This must be called first
        return self.param_dict[parameter_id][parameter_set]

    def get_parameter_sets(self):
        sets = defaultdict(list)
        for s in self.get_settings():
            sets[s['parameter_id']].append(s['parameter_set'])
        return list(sets.items())

class Task(object):

    def __init__(self, experiment_name, experiment_id,
                 train, test,
                 parameter_id, parameter_set):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.train = train
        self.test = test
        self.parameter_id = parameter_id
        self.parameter_set = parameter_set

        self.priority_adjustment = 0

        self.grounded = False

        self.last_checkin = None
        self.finished = False
        self.failed = False
        self.in_progress = False

        self.finish_time = None

    def ground(self, results_directory,
               experiment_format, parameter_format):
        self.results_directory = results_directory
        self.experiment_format = experiment_format
        self.parameter_format = parameter_format

        self.parameter_id_str = ('_'.join(parameter_format)
                                 % self.parameter_id)
        self.experiment_id_str = ('_'.join(experiment_format)
                                  % self.experiment_id)

        results_subdir = os.path.join(self.results_directory,
                                      self.experiment_name)
        self.results_path = os.path.join(results_subdir,
                                         self.experiment_id_str + '.db')

        self.results_manager = get_result_manager(self.results_path)
        if self.results_manager.is_finished(self.train, self.test,
                self.parameter_id_str, self.parameter_set):
            self.finish()

        self.grounded = True

    def get_predictions(self, bag_or_inst, train_or_test):
        if not self.grounded:
            raise Exception('Task not grounded!')

        if not self.finished:
            raise UnfinishedException()

        if train_or_test == 'train':
            test_set_labels = False
        elif train_or_test == 'test':
            test_set_labels = True
        else:
            raise ValueError('"%s" neither "train" nor "test"' %
                             train_or_test)

        if bag_or_inst.startswith('b'):
            return self.results_manager.get_bag_predictions(
                self.train, self.test, self.parameter_id_str,
                self.parameter_set, test_set_labels)
        elif bag_or_inst.startswith('i'):
            return self.results_manager.get_instance_predictions(
                self.train, self.test, self.parameter_id_str,
                self.parameter_set, test_set_labels)
        else:
            raise ValueError('"%s" neither "bag" nor "instance"'
                             % bag_or_inst)

    def get_statistic(self, statistic_name):
        if not self.finished:
            raise UnfinishedException()

        return self.results_manager.get_statistic(statistic_name, self.train,
                    self.test, self.parameter_id_str, self.parameter_set)

    def store_results(self, submission):
        """Write results to disk."""
        if not self.grounded:
            raise Exception('Task not grounded!')

        self.results_manager.store_results(submission,
            self.train, self.test, self.parameter_id_str, self.parameter_set)

    def ping(self):
        if not self.finished:
            self.in_progress = True
            self.last_checkin = time.time()

    def quit(self):
        if not self.finished:
            self.in_progress = False
            self.last_checkin = None

    def fail(self):
        if not self.finished:
            self.failed = True
            self.in_progress = False

    def staleness(self):
        return time.time() - self.last_checkin

    def priority(self):
        return (10000*int(self.in_progress) + 1000*int(self.failed)
                + self.priority_adjustment)

    def finish(self):
        self.finished = True
        self.in_progress = False
        self.failed = False
        self.finish_time = time.time()

class ExperimentConfiguration(object):

    def __init__(self, experiment_name, experiment_id,
                 fold_config, param_config, resampling_constructor):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.fold_config = fold_config
        self.param_config = param_config
        self.resampling_constructor = resampling_constructor
        self.settings = None

    def get_settings(self):
        if self.settings is None:
            self.settings = []
            for train, test in self.fold_config.get_all_train_and_test():
                resampling_config = self.resampling_constructor(train)
                for resampled in resampling_config.get_all_resampled():
                    for pconfig in self.param_config.get_settings():
                        setting = {'train': resampled,
                                   'test': test}
                        setting.update(pconfig)
                        self.settings.append(setting)

        return self.settings

    def get_key(self, **setting):
        key = (self.experiment_name, self.experiment_id,
               setting['train'], setting['test'],
               setting['parameter_id'], setting['parameter_set'])
        return key

    def get_task(self, **setting):
        key = self.get_key(**setting)
        return Task(*key)

def start_experiment(configuration_file, results_root_dir):
    task_dict, param_dict = load_config(configuration_file, results_root_dir)

    server = ExperimentServer(task_dict, param_dict, render)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0'})
    cherrypy.quickstart(server)

def load_config(configuration_file, results_root_dir):
    tasks = {}
    parameter_dict = {}

    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    experiment_key = configuration['experiment_key']
    experiment_name = configuration['experiment_name']

    if experiment_name == 'mi_kernels':
        from resampling import NullResamplingConfiguration
        def constructor_from_experiment(experiment):
            return lambda dset: NullResamplingConfiguration(dset)
    else:
        raise ValueError('Unknown experiment name "%s"' % experiment_name)

    for experiment in configuration['experiments']:
        try:
            experiment_id = tuple(experiment[k] for k in experiment_key)
        except KeyError:
            raise KeyError('Experiment missing identifier "%s"'
                            % experiment_key)

        def _missing(pretty_name):
            raise KeyError('%s not specified for experiment "%s"'
                            % (pretty_name, str(experiment_id)))

        def _resolve(field_name, pretty_name):
            field = experiment.get(field_name,
                        configuration.get(field_name, None))
            if field is None: _missing(pretty_name)
            return field

        print 'Setting up experiment "%s"...' % str(experiment_id)

        try:
            dataset = experiment['dataset']
        except KeyError: _missing('Dataset')

        experiment_format = _resolve('experiment_key_format',
                                     'Experiment key format')

        parameter_key = _resolve('parameter_key', 'Parameter key')
        parameter_format = _resolve('parameter_key_format',
                                    'Parameter key format')
        parameters = _resolve('parameters', 'Parameters')
        param_config = ParameterConfiguration(results_root_dir,
                        experiment_name, experiment_id,
                        experiment_format, parameter_key,
                        parameter_format, parameters)
        parameter_dict[experiment_id] = param_config

        folds = _resolve('folds', 'Folds')
        fold_config = FoldConfiguration(dataset, *folds)

        resampling_constructor = constructor_from_experiment(experiment)

        priority = experiment.get('priority', 0)

        experiment_config = ExperimentConfiguration(
                                experiment_name, experiment_id,
                                fold_config, param_config,
                                resampling_constructor)
        settings = experiment_config.get_settings()
        prog = ProgressMonitor(total=len(settings), print_interval=10,
                               msg='\tGetting tasks')
        for setting in settings:
            key = experiment_config.get_key(**setting)
            task = experiment_config.get_task(**setting)
            task.priority_adjustment = priority
            task.ground(results_root_dir,
                experiment_format, parameter_format)
            tasks[key] = task
            prog.increment()

    return tasks, parameter_dict

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    start_experiment(*args, **options)
