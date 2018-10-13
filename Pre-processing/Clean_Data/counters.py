import numpy as np
import matplotlib.pyplot as plt

def countInvalid(data, invalidValue):
    '''Returns an array of length D, each elements counts how many invalidValue(s) there are in the column'''
    D = data.shape[1]
    counter = np.array( [ (data[:,i] == invalidValue).sum() for i in range(D) ] )
    return counter

def print_invalid(data,invalidValue):
    numInvalidValues = countInvalid(data,invalidValue)
    percentages=numInvalidValues/(data.shape[0])
    print("Number of invalid values per column:\n\r")
    print(numInvalidValues)
    print("\n\r")
    print("Percentage of invalid values per column :\n\r")
    print(percentages)
    return

