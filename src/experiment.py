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
from quadprog import quadprog
from mil_coordescent_boost_base_classifier import MIL_Base
from sklearn.metrics import hamming_loss

INSTANCE_PREDICTIONS = True

CLASSIFIERS = {
    'svm': SVM,
    'svr': MIKernelSVR,
    'embedded_svm' : EmbeddedSpaceSVM,
    'mil_base': MIL_Base,
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
    C_derivative_dict = task['C_derivative']
    print 'Starting task %s...' % str(experiment_id)
    print 'Training Set: %s' % train_dataset
    print 'Test Set:     %s' % test_dataset
    print 'Parameters:'
    for k, v in parameters.items():
        print '\t%s: %s' % (k, str(v))
    #print 'instance weights: %s' % instance_weights_dict.values()[:10]
    print 'bag C derivative: %s' % C_derivative_dict.values()[:10]
    #import pdb;pdb.set_trace()
   
    train = get_dataset(train_dataset)
    test = get_dataset(test_dataset)

    C_derivative_list = [C_derivative_dict[C_key] for C_key in train.bag_ids ]
    C_derivative = np.float32(np.array(C_derivative_list).reshape((-1, 1)))

    instance_weights_list = [instance_weights_dict[weight_key] for weight_key in train.instance_ids  ]
    instance_weights = np.array( instance_weights_list )  #instance_weights should be of type array in order to be used by sklearn module
    
    if instance_weights_dict.has_key(test.instance_ids[0]):
      	instance_weights_test_list = [instance_weights_dict[weight_key] for weight_key in test.instance_ids  ]
	instance_weights_test = np.array( instance_weights_test_list )  #instance_weights should be of type array in order to be used by sklearn module
    else:
   	instance_weights_test=None
	
    	
    
    #import pdb;pdb.set_trace()
 
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
    #import pdb;pdb.set_trace()
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
    #import pdb;pdb.set_trace()
    
    num_inst_train = len(train.instance_ids)
    num_bag_train = len(set([x[0] for x in train.instance_ids]))
    num_inst_per_bag = num_inst_train/num_bag_train
    train_bag_labels = train.bag_labels[:, 0]
    train_bag_labels=np.asmatrix(train_bag_labels).reshape((-1,1))

    train_bag_labels_pos_neg1=2*train_bag_labels-1
    #import pdb;pdb.set_trace()
    


    print 'Training...'
    timer.start('training')
    #import pdb;pdb.set_trace()
    
    classifier0.fit(train.instances, train_bag_labels_pos_neg1, train.instance_ids, C_derivative)


    timer.stop('training')
    #import pdb;pdb.set_trace()

    print 'Computing test bag predictions...'
    timer.start('test_bag_predict')
     
    instance_predictions0 = classifier0.predict_instance(test.instances)
    bag_predictions0 = classifier0.predict_bag(test.instances, test.instance_ids)

    test_bag_labels = bag_predictions0

    timer.stop('test_bag_predict')


    if INSTANCE_PREDICTIONS:
        print 'Computing test instance predictions...'
        timer.start('test_instance_predict')
        instance_predictions = instance_predictions0 
        timer.stop('test_instance_predict')


    print 'Computing train bag predictions...'
    timer.start('train_bag_predict')
    train_instance_predictions0 = classifier0.predict_instance(train.instances)
    train_bag_predictions0 = classifier0.predict_bag(train.instances, train.instance_ids)
    train_bag_labels = train_bag_predictions0
    
    timer.stop('train_bag_predict')

    if INSTANCE_PREDICTIONS:
    	print  'Computing train instance predictions...'
        timer.start('train_instance_predict')
        train_instance_labels = train_instance_predictions0 
        timer.stop('train_instance_predict')

    
    print 'Constructing submission...'
    # Add statistics
    for attribute in ('linear_obj', 'quadratic_obj'):
        if hasattr(classifier0, attribute):
            submission['statistics'][attribute] = getattr(classifier,
                                                          attribute)
    submission['statistics'].update(timer.get_all('_time'))
    #import pdb;pdb.set_trace()
    bag_predictions=bag_predictions0
    for i, y in zip(test.bag_ids, map(tuple, bag_predictions)):
        submission['bag_predictions']['test'][i] = map(float,y)


    train_bag_labels = train_bag_predictions0
    for i, y in zip(train.bag_ids, map(tuple, train_bag_labels)):
        submission['bag_predictions']['train'][i] = map(float,y)
    if INSTANCE_PREDICTIONS:
        for (b, i), y in zip(test.instance_ids, instance_predictions.flat):
            submission['instance_predictions']['test'][(b,i)] =float(y)
        for (b, i), y in zip(train.instance_ids, train_instance_labels.flat):
            submission['instance_predictions']['train'][(b, i)] = float(y)

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
    #import pdb;pdb.set_trace()
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
	    for ii in range(1):
		AUC_list.append(score(np.array(test.bag_labels[:,ii]), np.array(test_bag_labels[:,ii]))) #weighted AUC 
	    AUC_mean=np.mean(AUC_list)
	    submission['statistics'][scorename]=AUC_mean
	    print ('Test Bag Average %s Score: %f'
                   % (scorename,AUC_mean ))
	    print( 'Test Bag Individual %s Score: ' %scorename   +','.join(map(str, AUC_list))   )

            Accuracy_test = 1 - hamming_loss(test.bag_labels[:, 0], test_bag_labels>0)
            print( 'Test Bag accuracy: %f' % Accuracy_test  )


	    AUC_train=score(np.array(train.bag_labels[:,0]), np.array(train_bag_labels[:,0]) )
	    Accuracy_train = 1- hamming_loss(train.bag_labels[:, 0], train_bag_labels>0)
	    print ('Train Bag Average %s Score: %f'
                   % ('AUC', AUC_train ))
	    print( 'Train Bag accuracy: %f' % Accuracy_train  )
	
        """
        if INSTANCE_PREDICTIONS and test.instance_labels.size > 1:
            print ('Test Inst. %s Score: %f'
                   % (scorename, score(test.instance_labels, instance_predictions)))
        """
    except Exception as e:
        print "Couldn't compute scores."
        print e
    import pdb;pdb.set_trace()

    print 'Finished task %s.' % str(experiment_id)
    return submission
