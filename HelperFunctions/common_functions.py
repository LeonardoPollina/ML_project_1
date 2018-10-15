import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x
    

def split_data(x, y, ratio, seed=1):
    """
    TODO: GENERALIZE WHEN X IS A MATRIX
    
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing

    return x_tr, x_te, y_tr, y_te
    """
    # set seed
    np.random.seed(seed)
    xrand = np.random.permutation(x)
    
    np.random.seed(seed)
    yrand = np.random.permutation(y)
    
    limit = int(y.shape[0]*ratio)
    
    return xrand[:limit], xrand[limit:], yrand[:limit], yrand[limit:]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


