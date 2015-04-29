"""
Implements the actual client function to run the experiment
"""
import os
import numpy as np
import time
import math
from data import get_dataset
from mi_svm import MIKernelSVM, MIKernelSVR, SVM
from vocabulary import EmbeddedSpaceSVM

INSTANCE_PREDICTIONS = False

CLASSIFIERS = {
    'svm': SVM,
    'svr': MIKernelSVR,
    'embedded_svm' : EmbeddedSpaceSVM,
}

IDX_DIR = os.path.join('box_counting', 'converted_datasets')
PRECOMPUTED_DIR = os.path.join('box_counting', 'precomputed')
IDX_FMT = '%s.idx'
PRECOMPUTED_FMT = '%s_%s.db'

def get_base_dataset(train):
    parts = train.split('.')
    i = 1
    while not parts[i].startswith('fold_'):
        i += 1
    return '.'.join(parts[:i])

class Timer(object):

    def __init__(self):
        self.starts = {}
        self.stops = {}

    def start(self, event):
        self.starts[event] = time.time()

    def stop(self, event):
        self.stops[event] = time.time()

    def get(self, event):
        return self.stops[event] - self.starts[event]

    def get_all(self, suffix=''):
        times = {}
        for event in self.stops.keys():
            times[event + suffix] = self.get(event)
        return times

def client_target(task, callback):
    (experiment_name, experiment_id,
     train_dataset, test_dataset, _, _) = task['key']
    parameters = task['parameters']
    instance_weights_dict=task['instance_weights']
    print 'Starting task %s...' % str(experiment_id)
    print 'Training Set: %s' % train_dataset
    print 'Test Set:     %s' % test_dataset
    print 'Parameters:'
    for k, v in parameters.items():
        print '\t%s: %s' % (k, str(v))
    print 'instance weights: %s' % instance_weights_dict.values()[:10]
    #import pdb;pdb.set_trace()
   
    train = get_dataset(train_dataset)
    test = get_dataset(test_dataset)

    instance_weights_list = [instance_weights_dict[weight_key] for weight_key in train.instance_ids  ]
    instance_weights = np.array( instance_weights_list )  #instance_weights should be of type array in order to be used by sklearn module
    
    if instance_weights_dict.has_key(test.instance_ids[0]):
      	instance_weights_test_list = [instance_weights_dict[weight_key] for weight_key in test.instance_ids  ]
	instance_weights_test = np.array( instance_weights_test_list )  #instance_weights should be of type array in order to be used by sklearn module
    else:
   	instance_weights_test=None

    
    #import pdb;pdb.set_trace()
    """
    data_raw = np.genfromtxt('natural_scene.data',delimiter = ",")
    class data_class(object):
	def __init__(self):
		pass
    train=data_class()
    test=data_class()
    feature_matrix = data_raw[:, 2:-5]
    label_matrix = data_raw[:, -5:]
    num_instances = data_raw.shape[0]
    train.instances = feature_matrix[:int(math.floor(num_instances/2)),: ]
    test.instances = feature_matrix[int(math.floor(num_instances/2)):,: ]
    train.instance_labels = label_matrix[:int(math.floor(num_instances/2)),: ]
    test.instance_labels = label_matrix[int(math.floor(num_instances/2)):,:  ]
    """
    submission = {
        'instance_predictions' : {
            'train' : {},
            'test'  : {},
        },
        'bag_predictions' : {
            'train' : {},
            'test'  : {},
        },
        'statistics' : {}
    }
    timer = Timer()

    if parameters['kernel'] == 'emp':
        dataset = get_base_dataset(train_dataset)
        idxfile = os.path.join(IDX_DIR, IDX_FMT % dataset)
        kernelfile = os.path.join(PRECOMPUTED_DIR,
            PRECOMPUTED_FMT % (dataset, parameters['ktype']))
        parameters['dataset'] = dataset
        parameters['idxfile'] = idxfile
        parameters['kernelfile'] = kernelfile
        empirical_labels = list(map(str, train.bag_ids))
        if parameters.pop('transductive', False):
            empirical_labels += list(map(str, test.bag_ids))
        parameters['empirical_labels'] = empirical_labels
        train.bags = train.bag_ids
        test.bags = test.bag_ids

    classifier_name = parameters.pop('classifier')
    if classifier_name in CLASSIFIERS:
        classifier0 = CLASSIFIERS[classifier_name](**parameters)
	classifier1 = CLASSIFIERS[classifier_name](**parameters)
	classifier2 = CLASSIFIERS[classifier_name](**parameters)
 	classifier3 = CLASSIFIERS[classifier_name](**parameters)
 	classifier4 = CLASSIFIERS[classifier_name](**parameters)
    else:
        print 'Technique "%s" not supported' % classifier_name
        callback.quit = True
        return

    print 'Training...'
    timer.start('training')
    if train.regression:
        classifier1.fit(train.bags, train.bag_labels)
    else:
        #import pdb;pdb.set_trace()
	classifier0.fit(train.instances, train.instance_labels[:,0].reshape((-1,)), instance_weights)
	classifier1.fit(train.instances, train.instance_labels[:,1].reshape((-1,)), instance_weights)
	classifier2.fit(train.instances, train.instance_labels[:,2].reshape((-1,)), instance_weights)
	classifier3.fit(train.instances, train.instance_labels[:,3].reshape((-1,)), instance_weights)
	classifier4.fit(train.instances, train.instance_labels[:,4].reshape((-1,)), instance_weights)
    timer.stop('training')

    print 'Computing test bag predictions...'
    timer.start('test_bag_predict')
    bag_predictions0 = classifier0.predict(test.instances)
    bag_predictions1 = classifier1.predict(test.instances)
    bag_predictions2 = classifier2.predict(test.instances)
    bag_predictions3 = classifier3.predict(test.instances)
    bag_predictions4 = classifier4.predict(test.instances)

    timer.stop('test_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing test instance predictions...'
        timer.start('test_instance_predict')
        instance_predictions = classifier.predict(test.instances_as_bags)
        timer.stop('test_instance_predict')

    print 'Computing train bag predictions...'
    timer.start('train_bag_predict')
    #train_bag_labels = classifier0.predict() # Saves results from training set

    train_bag_labels0 = classifier0.predict()
    train_bag_labels1 = classifier1.predict()
    train_bag_labels2 = classifier2.predict()
    train_bag_labels3 = classifier3.predict()
    train_bag_labels4 = classifier4.predict()

    timer.stop('train_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing train instance predictions...'
        timer.start('train_instance_predict')
        train_instance_labels = classifier.predict(train.instances_as_bags)
        timer.stop('train_instance_predict')

    print 'Constructing submission...'
    # Add statistics
    for attribute in ('linear_obj', 'quadratic_obj'):
        if hasattr(classifier0, attribute):
            submission['statistics'][attribute] = getattr(classifier,
                                                          attribute)
    submission['statistics'].update(timer.get_all('_time'))
    bag_predictions = np.hstack((bag_predictions0[:,np.newaxis], bag_predictions1[:,np.newaxis],bag_predictions2[:,np.newaxis],bag_predictions3[:,np.newaxis],bag_predictions4[:,np.newaxis]  ))
    for ( _,i), y in zip(test.instance_ids, map(tuple,bag_predictions)):
        submission['bag_predictions']['test'][i] = map(float,y)


    train_bag_labels = np.hstack((train_bag_labels0[:,np.newaxis], train_bag_labels1[:,np.newaxis],train_bag_labels2[:,np.newaxis],train_bag_labels3[:,np.newaxis],train_bag_labels4[:,np.newaxis]  ))
    for (_, i), y in zip(train.instance_ids, map(tuple,train_bag_labels)):
        submission['bag_predictions']['train'][i] = map(float,y)
    if INSTANCE_PREDICTIONS:
        for i, y in zip(test.instance_ids, instance_predictions.flat):
            submission['instance_predictions']['test'][i] =float(y)
        for i, y in zip(train.instance_ids, train_instance_labels.flat):
            submission['instance_predictions']['train'][i] = float(y)

    # For backwards compatibility with older versions of scikit-learn
    if train.regression:
        from sklearn.metrics import r2_score as score
        scorename = 'R^2'
    else:
        try:
            from sklearn.metrics import roc_auc_score as score
        except:
            from sklearn.metrics import auc_score as score
        scorename = 'AUC'

    try:
        """
        if train.bag_labels.size > 1:
            print ('Training Bag %s Score: %f'
                   % (scorename, score(train.instance_labels, train_bag_labels)))
        if INSTANCE_PREDICTIONS and train.instance_labels.size > 1:
            print ('Training Inst. %s Score: %f'
                   % (scorename, score(train.instance_labels, train_instance_labels)))
        """
        if test.bag_labels.size > 1:
            AUC_list=[]
	    for ii in range(5):
		AUC_list.append(score(test.instance_labels[:,ii], bag_predictions[:,ii], sample_weight = instance_weights_test)) #weighted AUC 
	    AUC_mean=np.mean(AUC_list)
	    submission['statistics'][scorename]=AUC_mean
	    print ('Test Bag Average %s Score: %f'
                   % (scorename,AUC_mean ))
	    print( 'Test Bag Individual %s Score: ' %scorename   +','.join(map(str, AUC_list))   )
        """
        if INSTANCE_PREDICTIONS and test.instance_labels.size > 1:
            print ('Test Inst. %s Score: %f'
                   % (scorename, score(test.instance_labels, instance_predictions)))
        """
    except Exception as e:
        print "Couldn't compute scores."
        print e

    print 'Finished task %s.' % str(experiment_id)
    return submission
