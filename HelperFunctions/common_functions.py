import numpy as np

def standardize(xOriginal, mean=np.array([]), std=np.array([])):
    """
    Standardize the original data set.
    Mean and std are the parameters computed on the training set.
    We compute the mean and the std of xOriginal, then standardize.
    """   
    if mean.size == 0 and std.size == 0:
        mean_x = np.mean(xOriginal,axis=0)
        x = xOriginal - mean_x
        std_x = np.std(x,axis=0)
        if (std_x==0).any():
            print('Substitute 0 with 0.0001')
            std_x[std_x == 0] = 1e-4
        x = x / std_x
    else:
        x = xOriginal - mean
        x = x / std
        mean_x,std_x = mean,std
    return x, mean_x, std_x

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]
    

def split_data(x, y, ratio, seed=1):
    """
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


