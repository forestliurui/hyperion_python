"""
Implement "Vocabulary-based" approaches
"""
import numpy as np
from scipy.spatial.distance import cdist

from mi_svm import SVM
from linear import L1SVM

class Vocabulary(object):

    def __init__(self, **parameters):
        # Options for the vocabulary
        # (e.g. distance metric, use all instances or clustering?)
        parameters = dict(**parameters)

        if 'vocab_similarity' not in parameters:
            raise ValueError("A value for vocab_similarity must be provided")
        self.similarity_type = parameters.pop('vocab_similarity')
        if self.similarity_type == 'miles':
            self.similarity = lambda x: np.min(x, axis=0)
        elif self.similarity_type == 'yards':
            self.similarity = lambda x: np.average(x, axis=0)
        else:
            raise ValueError('invalid value "%s" for vocab_similarity'
                             % self.similarity_type)

        if 'vocab_type' not in parameters:
            raise ValueError("A value for vocab_type must be provided")
        self.vocab_type = parameters.pop('vocab_type')
        if self.vocab_type not in ['instances']:
            raise ValueError("invalid value for vocab_type")

        if 'vocab_gamma' not in parameters:     #  May not always be needed
            raise ValueError("A value for vocab_gamma must be provided")
        self.gamma = parameters.pop('vocab_gamma')

    def fit(self, X):
        # Given a set of instances X, construct the
        # vocabulary (e.g. perform clustering, or just
        # store the instances)
        if self.vocab_type == 'instances':
            self.vocabulary = X
            return X
        else:
            assert False
        #  Insert K-Means, others here        

    def transform(self, B):
        # Transform the list of bags B to an array
        # of feature vectors (rows) for each bag
        #s(k, Bi) = maxj exp(-gamma*||xij - xk||^2)
        embeddings = [self.similarity(
                        np.exp(-self.gamma
                            * cdist(bag, self.vocabulary, 'sqeuclidean')))
                      for bag in B]
        return np.vstack(embeddings)

class EmbeddedSpaceSVM(object):

    def __init__(self, **parameters):
        if 'regularization' not in parameters:
            raise ValueError("A value for regularization must be provided")
        self.regularization = parameters.pop('regularization')
        if self.regularization not in ["L1", "L2"]:
            raise ValueError("invalid value for regularization")

        self.vocabulary = Vocabulary(**parameters)

        if self.regularization == "L1":
            self.svm = L1SVM(**parameters)
        elif self.regularization == "L2":
            self.svm = SVM(**parameters)

    def fit(self, B, y):
        # Make sure B, y are properly formatted arrays
        y = 2.0*(np.asarray(y) == 1) - 1.0
        B = map(np.asarray, B)

        # Collect all instances for vocabulary
        X = np.vstack(B)
        self.vocabulary.fit(X)
        features = self.vocabulary.transform(B)

        self.svm.fit(features, y)

    def predict(self, B=None):
        if B is None:
            return self.svm.predict()
        else:
            B = map(np.asarray, B)
            features = self.vocabulary.transform(B)
            return self.svm.predict(features)
