import numpy as np
def lsq(X,y):
    #calculates vector w from input matrix X and output values y
    
    #add a column of ones
    ones = np.ones((len(X),1))
    X = np.concatenate((ones,X), axis=1)
    
    #compute w
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    return w

def predict(X,w):
    #predicts output values based on an input matrix and vector w
    y_predicted = np.dot(np.concatenate((np.ones((len(X),1)),X), axis=1),w)
    return y_predicted
    
def msq(y_truth,y_predicted):
    #calculates the mean square error of a true y and predicted y
    error = np.mean(np.square(np.subtract(y_truth,y_predicted)))
    return error