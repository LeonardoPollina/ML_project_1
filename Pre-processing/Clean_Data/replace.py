import numpy as np
import matplotlib.pyplot as plt

def replaceWithMean(data,invalidValue,idxCols):
    '''Replace the invalidValue with the mean value of the column which it belongs, only in the columns indexed by idxCols. '''
    D = data.shape[1]
    for i in idxCols:
        idxValid = np.where( data[:,i] != invalidValue )
        idxInvalid = np.where( data[:,i] == invalidValue )
        mean = np.mean( data[idxValid,i] )
        data[idxInvalid,i] = mean
    return data

def replaceWithZero(data,invalidValue,idxCols):
    '''Replace the invalidValue with 0, only in the columns indexed by idxCols. '''
    D = data.shape[1]
    for i in idxCols:
        idxInvalid = np.where( data[:,i] == invalidValue)
        data[idxInvalid,i] = 0
    return data