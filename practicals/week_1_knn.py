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
        [list] -- [of length(k). Contains the data for the k closest 
        neighbours of testInstance. ]
        [list] -- of length k. Contains the distances of the closest k 
        neighbours of TestInstance.
    """
    trainData = np.concatenate((X_train, y_train), axis=1)          #creates combined matrix of data and label

    
    distances = []
    for i in range(len(trainData)):                                 #calculate distance from testInstance to every trainDatapoint
        dist = euclidean_distance(testInstance, trainData[i][:np.shape(X_train)[1]])
        distances.append((trainData[i,:], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):                                              #get the k-nearest
        neighbours.append(distances[x][0])
    return neighbours

def get_response(neighbours):
    """[This function returns the label that is most occurent.]
    
    Arguments:
        neighbours {[list]} -- [Each element contains the data with as last value the label.]
    
    Returns:
        [integer / string] -- [label. Depending on the data is this a integer or string. ]
    """
    classvotes = {}
    for i in range(len(neighbours)):
        response = neighbours[i][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1
    
    sortedVotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def perform_knn(X_train, y_train, X_test, k, regression):
    """Creates testData, which is a matrix of X_test, with the predicted labels based
    on k nearest neighbours. 
    
    Arguments:
        X_train {[matrix]} -- [contains all training data]
        y_train {[vector]} -- [contains the labels of the training data]
        X_test {[matrix]} -- [contains all testing data]
        k {[integer]} -- [number of neighbours to take into account]
    
    Returns:
        [matrix] -- [X_test with the predicted labels or predicted values 
        dependent on classification or regression]
    """
    if regression!=True:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    elif regression:
        prediction=np.zeros((np.shape(X_test)[0],np.shape(X_test)[1]))

    testData = np.concatenate((X_test, np.zeros((len(X_test),1))), axis=1)  #initialize testData with empty labels, to replace later with predicted labels. 

    for i in range(len(X_test)):
        neighbours = get_neighbours(X_train, y_train, X_test[i], k)
        if regression:
            predict=np.zeros((np.shape(X_train)[1]))
            for neig in neighbours:
                predict += neig[:np.shape(X_train)[1]]
            prediction[i,:]=predict/k
        else:
            predicted_label = get_response(neighbours)
            testData[i][-1] = predicted_label    
                               #replace the zerolabel in testData with the predicted label
    if regression:
        return prediction
    else:
        return testData

def mse(truth,predict):
    '''Calculate the mean squared errors when cosidering multiple dimensions'''
    return np.mean(np.mean(np.square(np.subtract(truth,predict)),axis=1))

def regression_plot():
    '''Create a plot to show the MSE of the kNN regression algorithm for 
    different k values'''
    
    #download the necessary data
    diabetes = load_diabetes()

    X_train = diabetes.data[:300, :]
    y_train = diabetes.target[:300, np.newaxis]
    X_test = diabetes.data[300:, :]
    

    a=25
    errors=np.zeros(a)
    for idx, k in enumerate(range(1,2*a,2)):
        prediction=perform_knn(X_train, y_train, X_test, k, True)
        errors[idx]=mse(X_test,prediction)
    plt.plot(range(1,2*a,2),errors)
    plt.xlabel('k (Number of considered neighbours)')
    plt.ylabel('Mean Squared Error (-)')
    plt.title('MSE for k Nearest Neighbour regression')
    plt.show()
    
    return

def get_accuracy(testData, y_test):
    prediction = testData[:,-1]
    correct = np.count_nonzero(prediction==y_test[:,0])
    return correct    

def plot_error(testData, y_test, k):
    breast_cancer = load_breast_cancer()

    X_train = breast_cancer.data[:300, :]
    y_train = breast_cancer.target[:300, np.newaxis]
    X_test = breast_cancer.data[300:, :]
    y_test = breast_cancer.target[300:, np.newaxis] 
    return


