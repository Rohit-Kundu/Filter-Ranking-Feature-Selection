import numpy as np
from sklearn import datasets

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

# Mean Absolute Dispersion
def MAD(data,target):
    mean_abs_diff = np.sum(np.abs(data-np.mean(data,axis=0)),axis=0)/data.shape[0]
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = mean_abs_diff
    result.ranks = np.argsort(np.argsort(mean_abs_diff))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = MAD(data.data, data.target)
