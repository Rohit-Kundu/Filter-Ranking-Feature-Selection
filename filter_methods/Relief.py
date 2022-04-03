import numpy as np
from sklearn import datasets
from ReliefF import ReliefF

def normalize(vector, lb=0, ub=1):
    # function to normalize a numpy vector in [lb, ub]
    norm_vector = np.zeros(vector.shape[0])
    maximum = max(vector)
    minimum = min(vector)
    norm_vector = lb + ((vector - minimum)/(maximum - minimum)) * (ub - lb)

    return norm_vector

class Result():
    # structure of the result
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None 

# Relief
def Relief(data, target):
    # function that assigns scores to features according to Relief algorithm
    # the rankings should be done in increasing order of the Relief scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    result = Result()
    result.features = feature_values

    # generate the ReliefF scores
    relief = ReliefF(n_neighbors=5, n_features_to_keep=num_features)
    relief.fit_transform(data, target)
    result.scores = normalize(relief.feature_scores)
    result.ranks = np.argsort(np.argsort(-relief.feature_scores))

    # produce scores and ranks from the information matrix
    Relief_scores = normalize(relief.feature_scores)
    Relief_ranks = np.argsort(np.argsort(-relief.feature_scores))

    # assign the results to the appropriate fields
    result.scores = Relief_scores
    result.ranks = Relief_ranks
    result.ranked_features = feature_values[:, Relief_ranks]

    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = Relief(data.data, data.target)
