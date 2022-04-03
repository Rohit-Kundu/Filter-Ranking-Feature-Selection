import numpy as np
from sklearn.feature_selection import chi2
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

def chi_square(data, target):
    data = data.clip(min=0)
    chi,_ = chi2(data, target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = chi
    result.ranks = np.argsort(np.argsort(chi))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = chi_square(data.data, data.target)
