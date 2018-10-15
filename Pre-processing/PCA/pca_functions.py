import numpy as np
import matplotlib.pyplot as plt    

def PCAWithCovariance(x):
    '''x must have 0 mean. Returns the variance for each feature in percentage and
    the matrix containing the eigenVectors (i.e the combinations that gives 
    principal components).
    
    return percVariance, eVectors '''
    
    #covariance matrix
    COV = np.cov(x, rowvar = False)

    #ordered eValues/eVectors of COV
    eValues, eVectors = np.linalg.eig(COV)

    #contributions of each variables to the variance
    totVariance = sum(eValues)
    percVariance = eValues/totVariance

    return percVariance, eVectors


def PCAWithScatterMatrix(x):
    '''x must have 0 mean. Returns the variance for each feature in percentage and
    the matrix containing the eigenVectors (i.e the combinations that gives 
    principal components).
    
    return percVariance, eVectors '''
    
    #scatter matrix
    XX = x.T @ x

    #ordered eValues/eVectors of COV
    eValues, eVectors = np.linalg.eig(XX)

    #contributions of each variables to the variance
    totVariance = sum(eValues)
    percVariance = eValues/totVariance

    return percVariance, eVectors


def showCumulativeVariance(percVariance1, percVariance2):

    cumulative1 = np.cumsum(percVariance1)
    fig = plt.figure(figsize=[10,5])
    ax1 = plt.subplot(1,2,1)
    ax1 = plt.plot(cumulative1)
    ax1 = plt.title("Cumulative variance with method 1")

    cumulative2 = np.cumsum(percVariance2)
    ax2 = plt.subplot(1,2,2)
    ax2 = plt.plot(cumulative2)
    ax2 = plt.title("Cumulative variance with method 2")

    print('First method gives:\n')
    print(cumulative1)
    print('Second method gives:\n')
    print(cumulative2)

    return