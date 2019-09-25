import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
import matplotlib.pyplot as plt
import operator

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1-vector2)

def normalize(matrix):
    return matrix / matrix.max(axis=0)

def get_neighbours(X_train, y_train, testInstance, k):
    """[Function that calculates the euclidean distance from testInstance to 
    every point in X_train. The k closest neighbours are then returned,
    to perform KNN classification on. ]
    
    Arguments:
        X_train {[matrix]} -- [contains all training data]
        y_train {[vector]} -- [contains the labels of the training data]
        testInstance {[vector]} -- [data to be made prediction on]
        k {[integer]} -- [number of neighbours to take into account]
    
    Returns:
        [array] -- [of length(k). Contains the labels for the k closest 
        neighbours of testInstance. ]
    """

    distances = []
    for i in range(len(X_train)):                                 #calculate distance from testInstance to every trainDatapoint
        dist = euclidean_distance(testInstance, X_train[i,:])
        distances.append((y_train[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = [int(dist[0]) for dist in distances[0:k]]             #get the k-nearest
    return neighbours

def perform_knn(X_train, y_train, X_test, k, regression=False):
    """Performs KNN on train data, and returns the predicted labels
    
    Arguments:
        X_train {[matrix]} -- [contains all training data]
        y_train {[vector]} -- [contains the labels of the training data]
        X_test {[matrix]} -- [contains all testing data]
        k {[integer]} -- [number of neighbours to take into account]
    
    Returns:
        [array] -- [predicted labels for test data X_test]
    """
    
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    y_predict=np.zeros((np.shape(X_test)[0],1))

    for i in range(len(X_test)):
        neighbours = get_neighbours(X_train, y_train, X_test[i], k)
        if regression:        
            y_predict[i]=np.mean(neighbours)
        else:
            y_predict[i] = np.round(np.mean(neighbours))  
                               #replace the zerolabel in testData with the predicted label

    return y_predict

def plot_knn_performance(X_train, y_train, X_test, y_test, maxK=30):
    a = int(maxK/2)
    trues = np.zeros(a)
    for i, k in enumerate(range(1,maxK,2)):
        y_predict = perform_knn(X_train, y_train, X_test, k, False)
        trues[i]=np.sum(np.mean(y_predict==y_test)) 
    plt.plot(range(1,maxK,2),trues)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuray for KNN')
    plt.show()

def mse(truth,predict):
    '''Calculate the mean squared errors when cosidering multiple dimensions'''
    return np.mean(np.mean(np.square(np.subtract(truth,predict)),axis=1))

def regression_plot():
    '''Create a plot to show the MSE of the kNN regression algorithm for 
    different k values'''
    
    #download the necessary data
    diabetes = load_breast_cancer()

    X_train = diabetes.data[:300, :]
    y_train = diabetes.target[:300, np.newaxis]
    X_test = diabetes.data[300:, :]
    y_true = diabetes.target[300:, np.newaxis]

    a=25
    errors=np.zeros(a)
    for idx, k in enumerate(range(1,2*a,2)):
        prediction=perform_knn(X_train, y_train, X_test, k, True)
        errors[idx]=mse(y_true,prediction)
    plt.plot(range(1,2*a,2),errors)
    plt.xlabel('k (Number of considered neighbours)')
    plt.ylabel('Mean Squared Error (-)')
    plt.title('MSE for k Nearest Neighbour regression')
    plt.show()
    
    print('Minimum MSE: ' + str(np.min(errors)))
    return   
