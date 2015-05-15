import data
import numpy as np
import scipy.stats as ss
from sklearn.metrics import hamming_loss

class EvaluationMetric(object):
	def __init__(self, task, prediction_matrix_test_accumulated):
		test=data.get_dataset(task.test)
		'''
		prediction_inst=task.get_predictions('bag','test') #this is dictionary, with key as inst_id and value as list of scores for each label
        	self.prediction_matrix=reduce( lambda x, y :np.vstack((x, y)), [prediction_inst[x[1]]  for x in test.instance_ids   ]  )
		'''

		self.prediction_matrix=prediction_matrix_test_accumulated
        	self.label_matrix=test.instance_labels 

	def one_error(self):
		return 1-np.mean([ self.label_matrix[i, np.argmax(self.prediction_matrix[i,:])]  for i in range(self.prediction_matrix.shape[0]) ])

	def coverage(self):
        	#compute coverage      
        	max_rank_inst=[]
        	for i in range(self.prediction_matrix.shape[0]):
			max_rank_inst.append( max([ ss.rankdata(-self.prediction_matrix[i,:])[x]  for x in range(self.prediction_matrix.shape[1]) if self.label_matrix[i, x] == True ] )  )

        	#np.mean(max_rank_inst)
        	#import pdb;pdb.set_trace()
		return np.mean(max_rank_inst)-1
	
	def hamm_loss(self):
		prediction_matrix_boolean=(self.prediction_matrix>0)
		return np.mean([ hamming_loss(prediction_matrix_boolean[i,:], self.label_matrix[i,:] ) for i in  range(self.prediction_matrix.shape[0])  ] )
	
	def avg_prec(self):
		#import pdb;pdb.set_trace()
		avgprec_list=[]
		for i in range(self.prediction_matrix.shape[0]):
 			for j in range(self.prediction_matrix.shape[1]):
				temp_ins=[]
				if (self.label_matrix[i, j] == True  ):
					temp_ins.append( len([x for x in range(self.prediction_matrix.shape[1]) if self.label_matrix[i, x] == True and self.rank_f(i, x) <= self.rank_f(i, j) ])/float(self.rank_f(i, j))  )
			if not len(temp_ins)==0:
				avgprec_list.append( np.mean(temp_ins) )
		return np.mean(avgprec_list)

	def rank_f(self, instance_index, label_index):
        	#compute the rank of label with label_index (start from 0) for instance with instance_index (start from 0) in predictiom matrix
		#the highest with rank 1
		return ss.rankdata(-self.prediction_matrix[instance_index, :])[label_index]