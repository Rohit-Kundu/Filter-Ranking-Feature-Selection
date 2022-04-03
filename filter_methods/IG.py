import numpy as np
from sklearn.feature_selection import *
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

# Information Gain
def IG(data,target):
    importances = mutual_info_classif(data,target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = importances
    result.ranks = np.argsort(np.argsort(importances))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = IG(data.data, data.target)
