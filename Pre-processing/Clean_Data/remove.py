import numpy as np
import matplotlib.pyplot as plt

from counters import *

def removeLines(data, y, idxCol, invalidValue):
    '''Remove the lines in data that contains invalidValue in position idxCol.
    Note that we need to remove the elements also from the vector y, to be consistent.'''
    idx = np.where(data[:,idxCol] == invalidValue)
    data = np.delete(data,idx,axis=0)
    yret = np.delete(y, idx, axis=0)
    return data, yret

def removeColumns(data,threshold):
    ''' Remove the columns containing more than threshold (in %) of invalid value.'''
    ''' Threshold has to be a value between 0 and 1'''
    numInvalidValues = countInvalid(data,-999)
    numInvalidValues_percentage=numInvalidValues/(data.shape[0])
    idx_to_remove=np.where(numInvalidValues_percentage>threshold)
    data=np.delete(data,idx_to_remove,axis=1)
    return data