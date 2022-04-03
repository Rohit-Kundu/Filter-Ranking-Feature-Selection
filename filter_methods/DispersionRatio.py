import numpy as np
from sklearn import datasets
import warnings

warnings.filterwarnings("ignore")

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

# Dispersion Ratio
def DispersionRatio(data,target):
    data[np.where(data==0)[0]]=1
    am = np.mean(data,axis=0)
    gm = np.power(np.prod(data,axis=0),1/data.shape[0])
    disp_ratio = am/gm
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = disp_ratio
    result.ranks = np.argsort(np.argsort(disp_ratio))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = DispersionRatio(data.data, data.target)
