# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

plt.close("all")
breast_cancer = load_breast_cancer()

#Extract data
X = breast_cancer.data
print(X.shape)
Y = breast_cancer.target[:, np.newaxis]

#Normalize data to [0,1] range
X_norm = (X-X.min(axis = 0))/(X.max(axis = 0)-X.min(axis = 0))

#Set boundaries for the gaussian distributions and create subplots
x = np.linspace(norm.ppf(0.05), norm.ppf(0.95), 100)
fig, axs = plt.subplots(5, 6, constrained_layout=True)

#Draw subplots for each feature
for feat in range(X.shape[1]):
    feat_data = X_norm[:,np.newaxis,feat]
    #Extract data with corresponding label 0
    axs.flat[feat].plot(x, norm.pdf(x,feat_data[Y==0].mean(),feat_data[Y==0].std()),'r-', lw=2, alpha=0.6, label='norm pdf')
    #Extract data with corresponding label 1
    axs.flat[feat].plot(x, norm.pdf(x,feat_data[Y==1].mean(),feat_data[Y==1].std()),'b-', lw=2, alpha=0.6, label='norm pdf')
    axs.flat[feat].set_title(str(feat+1))
    axs.flat[feat].set_xlabel("x")
    axs.flat[feat].set_ylabel("P")

#Swap the axes of X to make it compatible with the Y vector
X_norm = np.swapaxes(X_norm,0,1)

#Compute the probability of X and Y occuring together
PXY = X_norm.dot(Y)/X_norm.shape[1]

#Compute the probability of Y
PY = Y.sum()/Y.shape[0]

#Compute the probability of Y, provided that X holds
P = PXY/PY
