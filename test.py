from sklearn import datasets
from filter_methods import *

iris = datasets.load_iris()
result = DispersionRatio(iris.data, iris.target)
print("Feature Ranks:\n",result.ranks)
