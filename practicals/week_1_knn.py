import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()

X_train = breast_cancer.data[:300, np.newaxis]
y_train = breast_cancer.target[:300, np.newaxis]
X_test = breast_cancer.data[300:, np.newaxis]
y_test = breast_cancer.target[300:, np.newaxis]

def euclidean_distance(vector1, vector2):
    """[summary]
    
    Arguments:
        vector1 {[type]} -- [description]
        vector2 {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.sqrt(np.sum(np.power(vector1-vector2, 2),axis=1))

def get_neighbours(X_train, X_test, k):
    """[summary]
    
    Arguments:
        X_train {[type]} -- [description]
        X_test {[type]} -- [description]
        k {[type]} -- [description]
    """
    distances = []
    for i in range(len(X_test[:,0,0])):
        distance = euclidean_distance(X_test[i,:,:],X_train)
        distances.append(distance)
    
    neighbours = []
    return distances
    