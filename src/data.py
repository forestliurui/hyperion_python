"""Utility for loading datasets and folds"""
import os
import numpy as np
from scipy.io import loadmat
from collections import defaultdict

from c45data import parse_c45

DATA_DIR = 'data'
VIEWS_PATH = [os.path.join(DATA_DIR, 'views')]
SIVAL_DATA = 'sival.mat'
VIEW_SUFFIX = '.view'

class LRUDatasetCache(object):

    def __init__(self, size):
        self.size = size

        self.queue = []
        self.cache = {}

    def contains(self, key):
        return key in self.cache

    def get(self, key):
        # Move to front of queue
        self.queue.remove(key)
        self.queue.append(key)
        return self.cache[key]

    def add(self, key, value):
        # Check if we need to evict
        if len(self.queue) >= self.size:
            evicted = self.queue.pop(0)
            self.cache.pop(evicted)
        self.queue.append(key)
        self.cache[key] = value

CACHE = LRUDatasetCache(25)

class MIDataset(object):
    """
    Stores relevant views of MI dataset
    """

    def __init__(self, ids, X, y, regression):
        self.regression = regression
        #self.instance_ids_temp = ids  #this is a tuple list, i.e. each element in instance_ids is (bag_id, instance_id)
        #self.instances_temp = X
        self.instances_as_bags = [xx.reshape((1, -1)) for xx in X]
        #self.instance_labels_temp = y
        if not regression:
            self.pm1_instance_labels = (2.0*y - 1)
        self.instance_dict = dict()
	#import pdb; pdb.set_trace()
        # Group instances into bags
        bag_id_dict = defaultdict(list)
        bag_dict = defaultdict(list)
        if regression:
            bag_label_dict = dict()
        else:
            bag_label_dict = defaultdict(lambda: False)
        bag_inst_label_dict = defaultdict(list)
        for (bid, iid), ex, yi in zip(ids, X, y):
            self.instance_dict[bid, iid] = (ex, yi)
            bag_id_dict[bid].append(iid)
            bag_dict[bid].append(ex)
            if regression:
                if bid in bag_label_dict:
                    if bag_label_dict[bid] != yi:
                        raise ValueError(
                            'Regression bag label mismatch in %s: %f != %f'
                            % (bid, yi, bag_label_dict[bid])
                        )
                else:
                    bag_label_dict[bid] = yi
            else:
                bag_label_dict[bid] |= yi  # "|=" means element-wise "or" between one-dimensional array "bag_label_dict[bid]" and one-dimensional array "yi" and 
            bag_inst_label_dict[bid].append(yi)
	
	self.bag_id_dict = bag_id_dict # self.bag_id_dict[bid] is a list of the iid contained in the bag bid
        self.bag_ids = sorted(bag_dict.keys())
        self.bags = [np.vstack(bag_dict[bid]) for bid in self.bag_ids]
        self.bag_labels = np.vstack([bag_label_dict[bid]
                                    for bid in self.bag_ids])
	self.instances = np.vstack(self.bags)
	self.instance_ids=[]
	self.instance_labels=[]
	
	for bid in self.bag_ids:
		for iid in bag_id_dict[bid]:
			self.instance_ids.append(  (bid, iid)  )
		#import pdb;pdb.set_trace()
		self.instance_labels.append( bag_inst_label_dict[bid]  )
	self.instance_labels=np.vstack(self.instance_labels)

        if not regression:
            self.pm1_bag_labels = (2.0*self.bag_labels - 1)
        self.bag_dict = dict()
        for bid, bi, yi in zip(self.bag_ids, self.bags, self.bag_labels):
            self.bag_dict[bid] = (bag_id_dict[bid], bi, yi, bag_inst_label_dict[bid])

class ViewBuilder(object):
    """
    Makes it easier to construct views in memory
    before writing them out to disk.
    """

    def __init__(self, name, dataset=None):
        self.name = name
        self.dataset = dataset
        self._lines = []

    def add(self, bag_id, instance_id=None,
            new_bag_id=None, new_instance_id=None, new_label=None):
        if self.dataset is None:
            raise Exception('Must call "add_from" or specify dataset.')

        self.add_from(self.dataset, bag_id, instance_id,
                      new_bag_id, new_instance_id, new_label)

    def add_from(self, dataset, bag_id, instance_id=None,
                 new_bag_id=None, new_instance_id=None, new_label=None):

        line = [dataset, bag_id]
        if instance_id is not None:
            line.append(instance_id)
            expect_new_bag_id = True
            expect_new_iid = True
        else:
            expect_new_bag_id = None
            expect_new_iid = False

        if new_bag_id is not None:
            line.append(new_bag_id)
        else:
            if expect_new_bag_id:
                raise ValueError('New bag id must be specified'
                                 ' with new instance id.')

        if (new_instance_id is None) != expect_new_iid:
            if new_instance_id is not None:
                line.append(new_instance_id)
        else:
            raise ValueError('New instance id must be specified'
                             ' if and only if old instance id is.')

        if new_label is not None:
            line.append(str(int(new_label)))

        line = ','.join(line)
        self._lines.append(line)

    def save(self, directory):
        path = os.path.join(directory, self.name + VIEW_SUFFIX)
        with open(path, 'w+') as f:
            f.write(str(self))

    def __str__(self):
        return '\n'.join(self._lines)

def get_dataset(dataset_name):
    if not CACHE.contains(dataset_name):
        try:
            dset = _get_base_dataset(dataset_name)
        except ValueError:
            # Not a base dataset
            dset = _get_data_view(dataset_name)

        CACHE.add(dataset_name, dset)

    return CACHE.get(dataset_name)

def does_view_exist(view_dataset_name):
    try:
        _find_view_file(view_dataset_name)
        return True
    except:
        return False

def _find_view_file(dataset_name):
    view_filename = dataset_name + VIEW_SUFFIX
    for directory in VIEWS_PATH:
        view_path = os.path.join(directory, view_filename)
        if os.path.exists(view_path):
            return view_path
    raise ValueError('Data file or view "%s" not found.' % dataset_name)

def _get_data_view(dataset_name):
    view_filename = _find_view_file(dataset_name)

    ids = []
    insts = []
    labels = []
    regression = False
    with open(view_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split(',')
            dset_name = parts.pop(0)
            dset = get_dataset(dset_name)
            regression |= dset.regression
            if len(parts) == 1:
                bid = parts[0]
                instances, binsts, _, ilabels = dset.bag_dict[bid]
                for iid, bi, iyi in zip(instances, binsts, ilabels):
                    ids.append((bid, iid))
                    insts.append(bi)
                    labels.append(iyi)
                continue

            elif len(parts) == 2:
                bid = parts[0]
                new_bid = parts[1]
                instances, binsts, _, ilabels = dset.bag_dict[bid]
                for iid, bi, iyi in zip(instances, binsts, ilabels):
                    ids.append((new_bid, iid))
                    insts.append(bi)
                    labels.append(iyi)
                continue

            elif len(parts) == 3:
                bid = parts[0]
                new_bid = parts[1]
                new_blabel = (int(parts[2]) == 1)
                instances, binsts, _, _ = dset.bag_dict[bid]
                for iid, bi in zip(instances, binsts):
                    ids.append((new_bid, iid))
                    insts.append(bi)
                    labels.append(new_blabel)
                continue

            elif len(parts) == 4:
                bid = parts[0]
                iid = parts[1]
                new_bid = parts[2]
                new_iid = parts[3]
                ex, yi = dset.instance_dict[bid, iid]
                ids.append((new_bid, new_iid))
                insts.append(ex)
                labels.append(yi)
                continue

            elif len(parts) == 5:
                if regression:
                    raise ValueError(
                        'Re-label views not implemented for regression.'
                    )
                bid = parts[0]
                iid = parts[1]
                new_bid = parts[2]
                new_iid = parts[3]
                new_label = (int(parts[4]) == 1)
                ex, _ = dset.instance_dict[bid, iid]
                ids.append((new_bid, new_iid))
                insts.append(ex)
                labels.append(new_label)
                continue

            else:
                raise ValueError('Invalid view line: %s' % line)
    #import pdb;pdb.set_trace()
    X = np.vstack(insts)
    y = np.vstack(labels)

    if len(set(ids)) != len(ids):
        counts = defaultdict(int)
        for i in ids:
            counts[i] += 1
        dups = [k for k, v in counts.items() if v > 1]
        print dups
        raise ValueError('Duplicate keys detected in %s!' % dataset_name)

    return MIDataset(ids, X, y, regression)

def _get_base_dataset(dataset_name):
    regression = False
    if dataset_name.startswith('sival'):
        ids, X, y = _get_sival_dataset(dataset_name[6:])
    elif dataset_name.startswith('mir_'):
        ids, X, y = _get_mir_dataset(dataset_name[4:])
        regression = True
    else:
        exset = parse_c45(dataset_name, DATA_DIR)
        #import pdb; pdb.set_trace()
	raw_data = np.array(exset.to_float())
        
	X = raw_data[:, 2:-5]
        y = (raw_data[:, -5:] == 1)
        """
	X = raw_data[:,2:-1]
	y = (raw_data[:,-1] == 1).reshape((-1,))	
	"""
	ids = [(ex[0], ex[1]) for ex in exset]
	#import pdb; pdb.set_trace()
    if len(set(ids)) != len(ids):
        counts = defaultdict(int)
        for i in ids:
            counts[i] += 1
        dups = [k for k, v in counts.items() if v > 1]
        print dups
        raise ValueError('Duplicate keys detected in %s!' % dataset_name)
    
    # Normalize
    mean = np.average(X, axis=0)
    std = np.std(X, axis=0)
    std[np.nonzero(std == 0.0)] = 1.0
    X = ((X - mean) / std)
   
    return MIDataset(ids, X, y, regression)

def _get_sival_dataset(dataset_name):
    mat = loadmat(os.path.join(DATA_DIR, SIVAL_DATA))
    class_id = None
    for i, name in enumerate(mat['class_names'], 1):
        if name.strip() == dataset_name:
            class_id = i
            break

    if class_id is None:
        raise Exception('Unknown SIVAL dataset: %s' % dataset_name)

    ids = [(str(i[0].strip()), str(i[1].strip()))
           for i in mat['instance_ids']]
    X = mat['X']
    y = (mat['y'] == class_id).reshape((-1,))
    return ids, X, y

def _get_mir_dataset(dataset_name):
    mat = loadmat(os.path.join(DATA_DIR, dataset_name), appendmat=True)
    raw = mat[dataset_name]
    bids = raw[:, 0].astype(int)
    ids = [(str(bid), str(i)) for i, bid in enumerate(bids, 1)]
    X = raw[:, 1:-1]
    y = raw[:, -1].reshape((-1,))
    return ids, X, y
