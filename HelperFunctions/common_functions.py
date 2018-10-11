import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    TODO: GENERALIZE WHEN X IS A MATRIX
    
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    xrand = np.random.permutation(x)
    
    np.random.seed(seed)
    yrand = np.random.permutation(y)
    
    limit = int(y.shape[0]*ratio)
    
    return xrand[:limit], xrand[limit:], yrand[:limit], yrand[limit:]


