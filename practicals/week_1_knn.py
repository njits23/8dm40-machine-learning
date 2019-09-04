import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()
X_train = breast_cancer.data[:300, np.newaxis, 3]
y_train = breast_cancer.target[:300, np.newaxis]
X_test = breast_cancer.data[300:, np.newaxis, 3]
y_test = breast_cancer.target[300:, np.newaxis]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))

def get_neighbours(X_train, X_test, k):
    distances = []
    neighbours = []
    