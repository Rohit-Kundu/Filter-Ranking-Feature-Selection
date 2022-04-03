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

# Fisher Score
def FisherScore(data,target):
    mean = np.mean(data)
    sigma = np.var(data)
    unique = np.unique(target)
    mu = np.zeros(shape=(data.shape[1],unique.shape[0]))
    n = np.zeros(shape=(unique.shape[0],))
    var = np.zeros(shape=(data.shape[1],unique.shape[0]))
    for j in range(data.shape[1]):
        for i,u in enumerate(unique):
            d = data[np.where(target==u)[0]]
            n[i] = d.shape[0]
            mu[j,i] = np.mean(d[:,j])
            var[j,i] = np.var(d[:,j])
    fisher = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        sum1=0
        sum2=0
        for i,u in enumerate(unique):
            sum1+=n[i]*((mu[j,i]-mean)**2)
            sum2+=n[i]*var[j,i]
        fisher[j] = sum1/sum2
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = fisher
    result.ranks = np.argsort(np.argsort(fisher))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    result = FisherScore(data.data, data.target)
