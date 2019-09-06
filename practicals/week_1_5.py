# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

#Extract data
X = breast_cancer.data
print(X.shape)
Y = breast_cancer.target[:, np.newaxis]

#Normalize data to [0,1] range
X_norm = (X-X.min(axis = 0))/(X.max(axis = 0)-X.min(axis = 0))
#Y_norm = (Y-Y.min())/(Y.max()-Y.min())

#Swap the axes of X to make it compatible with the Y vector
X_norm = np.swapaxes(X_norm,0,1)

#Compute the probability of X and Y occuring together
PXY = X_norm.dot(Y)/X_norm.shape[1]

#Compute the probability of Y
PY = Y.sum()/Y.shape[0]

#Compute the probability of Y, provided that X holds
P = PXY/PY