'''
implements the base classifier for mil coordescent boost
'''

import numpy as np
from numpy.linalg import inv
from sklearn.svm import SVC, NuSVR

import kernel

def beta4subgradient_max(alpha_current, K_tilde, num_instance_each_bag_train):
	K_tilde=np.matrix(K_tilde) #make sure it to be of matrix type
	alpha_current=np.matrix(alpha_current)
	beta_current=np.zeros((K_tilde.shape[1],1))
	max_each_bag=np.zeros((len(num_instance_each_bag_train), 1))
	
	for bag_index_temp in range(len(num_instance_each_bag_train)):
		if bag_index_temp ==0:
			max_each_bag[bag_index_temp]=np.max(  (  np.matrix(K_tilde[:,0: num_instance_each_bag_train[bag_index_temp]]).transpose()  )*alpha_current)
		else:
			max_each_bag[bag_index_temp]=np.max(  (np.matrix(K_tilde[:,sum(num_instance_each_bag_train[0:bag_index_temp]):sum( num_instance_each_bag_train[0:bag_index_temp+1])]).transpose())*alpha_current)	
	
	for bag_index_temp in range(len(num_instance_each_bag_train)):
		for inst_index_temp in range(num_instance_each_bag_train[bag_index_temp]):
			if bag_index_temp == 0:
				if (K_tilde[:, inst_index_temp].transpose())*alpha_current == max_each_bag[bag_index_temp]:
					beta_current[inst_index_temp]=1
					break
			else:
				if (K_tilde[:, int(sum(num_instance_each_bag_train[0:bag_index_temp]))+inst_index_temp ].transpose())*alpha_current == max_each_bag[bag_index_temp]:
					beta_current[int(sum(num_instance_each_bag_train[0:bag_index_temp]))+inst_index_temp]=1
					break
		#import pdb;pdb.set_trace()
		'''
		if bag_index_temp == 0:
			beta_current[0:int(num_instance_each_bag_train[bag_index_temp])]= (  beta_current[0:int(num_instance_each_bag_train[bag_index_temp])]  )/sum( beta_current[0:int(num_instance_each_bag_train[bag_index_temp])] )
		else:
			beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))]= (  beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))]  )/sum( beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))] )
		'''	
		#import pdb;pdb.set_trace()
	#import pdb;pdb.set_trace()
	
	beta_final=beta_current
	return beta_final


class MIL_Base(object):
	def __init__(self, **parameters):
		svm_params = {'kernel' : 'precomputed'}
        	if 'C' in parameters:
            		svm_params['C'] = parameters.pop('C')
        	self.estimator = SVC(**svm_params)

        	# Get kernel name and pass remaining parameters to kernel
        	kernel_name = parameters.pop('kernel')
        	self.kernel = kernel.by_name(kernel_name, **parameters)

	def fit(self, X, bag_labels, instance_ids, C_derivative):
		#X is the instance matrix with row 
		X = np.asarray(X)
        	self.fit_data = X
        	#  X is a list of arrays so applying asarray function to everything in that list
        	#  If you passed in a list of lists, if each bag is an array the asarray funciton just returns it
        	#  but it converts a list of lists to a numpy array.
        	#import pdb; pdb.set_trace()
		self.gram_matrix = self.kernel(X, X)

		bag_labels=np.matrix(bag_labels)
		bag_labels=bag_labels.reshape((-1, 1))	

		C_derivative=np.matrix(C_derivative)
		C_derivative=C_derivative.reshape((-1, 1))	
		
		#get self.a, self.b
		K=np.matrix(self.gram_matrix)
		num_instance=K.shape[0]
		num_bag = len(set([x[0] for x in instance_ids]))

		num_instance_each_bag=np.zeros((num_bag,1))
		inst_ids_total=[x[0] for x in instance_ids]
		bag_id_index=0
		for bag_id in set([x[0] for x in instance_ids]):
			num_instance_each_bag[bag_id_index]=sum([x==bag_id for x in inst_ids_total])
			bag_id_index=bag_id_index+1


		K_tilde= np.vstack((    K, np.ones((1, sum(num_instance_each_bag)) )    ))
		Lambda = 100000
		
		alpha_current=np.matrix(np.ones((num_instance+1, 1)))  #initialization of alpha to be ones
		import pdb; pdb.set_trace()
		for iteration in range(10):
			beta=beta4subgradient_max(alpha_current, K_tilde, num_instance_each_bag)
			
			beta=np.matrix(beta)
			beta_bool=(np.array(beta)==1)
			K_array=np.array(K)
			K_sub=np.matrix(K_array[ :, beta_bool.reshape((1,-1))[0] ])
			
			self.a=(-1/Lambda)*inv(K)*(K_sub*np.multiply(C_derivative, bag_labels))
			

			if sum(np.multiply(C_derivative, bag_labels))>0:
				self.b=-np.max(np.diag(K))
			else:
				self.b=np.max(np.diag(K))
			alpha_current=np.matrix(np.vstack((self.a, self.b)))
			#import pdb; pdb.set_trace()
			print ('For CCCP iteration No. %d' % iteration)
			print ('beta is')
			print beta
			print ('a and b is')
			print self.a, self.b
			import pdb; pdb.set_trace()
	def predict_instance(self, X = None):
		if X is None:
			gram_matrix = self.gram_matrix
		else:
			X = np.asarray(X)
			gram_matrix = self.kernel(X, self.fit_data) 

		gram_matrix=np.matrix(gram_matrix)
		a=np.matrix(self.a)
		b=np.matrix(self.b)
		
		return gram_matrix*a+b

	def predict_bag(self, X, instance_ids):
		
		instance_prediction = self.predict_instance(X)
		num_bag = len(set([x[0] for x in instance_ids]))
		bag_ids_duplicate = [x[0] for x in instance_ids]
		bag_id = []
		for i in range(len(bag_ids_duplicate)):
			if bag_ids_duplicate[i] not in bag_id:
				bag_id.append(bag_ids_duplicate[i])
		bag_prediction=-np.matrix(np.ones((num_bag,1)))
		for i in range(num_bag):
			#import pdb;pdb.set_trace()
			bag_prediction[i]=np.max(instance_prediction[ np.array(bag_ids_duplicate) == bag_id[i] ])
		return bag_prediction
