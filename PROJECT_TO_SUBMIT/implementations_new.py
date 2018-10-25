import numpy as np
import matplotlib.pyplot as plt
import csv
from proj1_helpers import *

################################################################################
#                        The requested methods:                                #
################################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss_MSE(y, tx, w)
        grad = compute_gradient_MSE(y, tx, w)
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

## we should merge these 2 functions above and below

def gradient_descent(y, tx, initial_w, max_iters, gamma, threshold = 1e-6, compute_loss = compute_loss_MSE,compute_gradient = compute_gradient_MSE):
    """Gradient descent algorithm.
    
    return w, loss"""
    # Define parameters to store w and loss
    loss0 = 0
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad
        if n_iter % 10 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
        if np.abs(loss - loss0) < threshold:
              return w, loss
        loss0 = loss
        
    return loss,w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    batch_size = 1
    for n_iter in range(max_iters):
        #get the mini batch
        batch = next( batch_iter(y, tx, batch_size) )
        minibatch_y, minibatch_tx = batch[0], batch[1]

        grad = compute_gradient_MSE(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad

        #for the loss we use the whole dataset
        loss = compute_loss_MSE(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

## we should merge these 2 functions above and below
def stochastic_gradient_descent( y, tx, initial_w, batch_size, max_iters, gamma,
         compute_loss=compute_loss_MSE, compute_gradient=compute_gradient_MSE):
    """SDG algorithm

    return ws, losses"""
    # Define parameters to store w and loss
    #ws = [initial_w] LP
    #losses = [] LP
    w = initial_w
    for n_iter in range(max_iters):
        batch = next(batch_iter(y, tx, batch_size))
        minibatch_y, minibatch_tx = batch[0], batch[1]
        grad = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
        # ws.append(w) LP
        #for the loss I use the whole dataset
        loss = compute_loss(y, tx, w)
        #losses.append(loss) LP
        if n_iter%10==0:
            print("Stochastic gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def least_squares(y, tx):
    """Compute the least squares solution."""

    #solve the normal equations
    w = np.linalg.solve(tx.T@tx, tx.T@y)

    e = y - tx@w
    loss = np.mean(e**2)/2
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Compute ridge regression."""
    
    w = np.linalg.solve(tx.T@tx + lambda_*2*y.shape[0]*np.eye(tx.shape[1]), tx.T@y.T)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss   


#                                     Logit                                    #

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    print(pred)
    CorrectedZero = 1e-7
    #loss = np.ones(tx.shape[0]).dot(np.log(1+np.exp(tx.dot(w))))-y.dot(tx.dot(w))
    loss = (1+y).T.dot(np.log(pred+CorrectedZero)) + (1 - y).T.dot(np.log(1 - pred+CorrectedZero))
    loss_correction_only = (1+y).T.dot(np.log(CorrectedZero*np.ones(tx.shape[0]))) + (1 - y).T.dot(np.log(CorrectedZero*np.ones(tx.shape[0])))
    print('logistic loss: ',- loss)
    print('logistic loss corection only: ',- loss_correction_only)
    return np.squeeze(- loss)


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_logistic_loss(y, tx, w)
    grad = calculate_logistic_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

#                           Regularized Logit                                 #

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    #num_samples = y.shape[0]
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

#                                     Newton                                   #

def newton(y, tx, w0, gamma, max_iter, tol=1e-6):
    
    losses = [0]
    
    for i in range(max_iter):
        
        pred = sigmoid(tx.dot(w0)) 
        
        # H = XT*S*X, compute S*X
        SX = np.zeros((tx.shape[0],tx.shape[1]))
        
        Snn = pred * ( 1 - pred)
                      
        #fast way to do matrix * diagonalmatrix
        SX = (tx.T * Snn).T
        
        H = tx.T @ SX

        pred = sigmoid(tx.dot(w0))
        
        grad = (tx.T) @ (pred - y)

        w = w0 - gamma * np.linalg.solve(H, grad)
    
        w0 = w
            
        if np.abs(np.linalg.norm(grad)) < tol:
            print(f'Exit Newton, small gradient. Iters = {i}, gradient = {np.linalg.norm(grad)}, loss = {calculate_logistic_loss(y, tx, w)}') 
            return w, calculate_logistic_loss(y, tx, w)
        

    return w, calculate_logistic_loss(y, tx, w)



def newton_regularized(y, tx, w0, gamma, lambda_, max_iter, tol=1e-6):
    
    
    for i in range(max_iter):
        
        pred = sigmoid(tx.dot(w0)) 
        
        # H = XT*S*X, compute S*X
        SX = np.zeros((tx.shape[0],tx.shape[1]))
        
        Snn = pred * ( 1 - pred)
                      
#         for row in range(tx.shape[0]):
#             SX[row,:] = Snn[row]*tx[row]
        #more clever!
        SX = (tx.T * Snn).T
        
        H = tx.T @ SX + np.eye(tx.shape[1])*lambda_
        
        grad = (tx.T) @ (pred - y) + 2*lambda_*w0

        w = w0 - gamma * np.linalg.solve(H, grad)
    
        w0 = w
            
        if np.abs(np.linalg.norm(grad)) < tol:
            print(f'Exit Newton, small gradient. Iters = {i}, gradient = {np.linalg.norm(grad)}, loss = {calculate_logistic_loss(y, tx, w)}') 
            return w, calculate_logistic_loss(y, tx, w)
        

    return w, calculate_logistic_loss(y, tx, w)

################################################################################
#                               Loss functions                                 #
################################################################################

def compute_loss_MSE(y, tx, w):
    """Calculate the loss using MSE."""

    e = y - tx.dot(w)
    return np.mean(e**2)/2

def compute_gradient_MSE(y, tx, w):
    """Calculate gradient of the loss MSE."""

    return (-1/y.shape[0]) * (tx.T) @ (y - tx @ w)

##do we really use these functions ???
def compute_gradient_MAE(y, tx, w):
    n = y.shape[0]
    grad = np.zeros(len(w))
    error = y - tx@w
    subgradient = np.sign(error)
    grad = - tx.T @ subgradient / n
    return grad

def calculate_mse(e):
    """Calculate the MSE for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the MAE for vector e."""
    return np.mean(np.abs(e))


def calculate_rmse(e):
    """Calculate the RMSE for vector e."""
    return np.sqrt(np.mean(e**2))

def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_mae(y, tx, w):
    """Calculate the MAE loss. """

    e = y - tx.dot(w)
    return calculate_mae(e)

def compute_loss_rmse(y, tx, w):
    """Calculate the RMSE loss. """

    e = y - tx.dot(w)
    return calculate_rmse(e)

#### ????


################################################################################
#                             Cross validation                                 #
################################################################################

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#cross validation with ridge regression (We can add other types of CV when we choose other
#working regressors)
def cross_validation_with_ridge(y, x, k_indices, lambda_, printSTD = False):
    """CV regression according to the splitting in train/test given by k_indices.
    
    The returned quantities are the average of the quantities computed in the single folds
    
    return w, loss_tr, loss_te"""
    
    folds = k_indices.shape[0]
    loss_tr = np.zeros(folds)
    loss_te = np.zeros(folds)
    accuracy = np.zeros(folds)

    if len( x.shape ) == 1:
        w_avg = 0
    else:
        w_avg = np.zeros(x.shape[1])
    
    for k in range(folds):
        
        #split the data in train/test
        idx = k_indices[k]
        yte = y[idx]
        if len( x.shape ) == 1:
            xte = x[idx]
        else:
            xte = x[idx,:]
        ytr = np.delete(y,idx,0)
        xtr = np.delete(x,idx,0)

        #regression
        w = ridge_regression(ytr,xtr,lambda_)

        #compute losses
        loss_tr[k] = compute_loss_rmse(ytr,xtr,w)
        loss_te[k] = compute_loss_rmse(yte,xte,w)

        #accuracy
        y_pred = predict_labels(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)  

    if printSTD:
        print(f'STD of test error: {np.std(loss_te)}')

    
    return np.mean(accuracy), np.mean(loss_tr), np.mean(loss_te)

def cross_validation_with_logistic(y, x, k_indices, gamma, lambda_):
    """CV regression according to the splitting in train/test given by k_indices.
    
    The returned quantities are the average of the quantities computed in the single folds
    
    return the accuracy"""
    
    folds = k_indices.shape[0]
    accuracy = np.zeros(folds)

    if len( x.shape ) == 1:
        w_avg = 0
    else:
        w_avg = np.zeros(x.shape[1])
    
    for k in range(folds):
        
        #split the data in train/test
        idx = k_indices[k]
        yte = y[idx]
        if len( x.shape ) == 1:
            xte = x[idx]
        else:
            xte = x[idx,:]
            
        ytr = np.delete(y,idx,0)
        xtr = np.delete(x,idx,0)

        #learning by gradient descent (with logistic)
        loss, w = learning_by_penalized_gradient(ytr, xtr, w, gamma, lambda_)
        #accuracy
        y_pred = predict_logistic_labels(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)  
   
    return np.mean(accuracy)

################################################################################
#                             Additional functions                             #
################################################################################


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

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


def grid_search(y, tx, w0, w1, compute_loss=compute_loss_MSE):
    """Algorithm for grid search.
    
    return losses"""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i,j] = compute_loss( y, tx, np.array([ w0[i],w1[j] ]) )        
    return losses


################################################################################
#                                Preprocessing                                 #
################################################################################


### Counters

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

### Remove

def removeConstantColumns(data):
    '''Remove columns which are constants from the data.
       
       Return data, idx_removed
    '''
    std = np.std(data, axis = 0)
    idx_removed = np.where(std==0)[0]
    if len(idx_removed >0 ):
        data = np.delete(data,idx_removed,axis=1)
    
    return data, idx_removed

def removeHighCorrelatedColumns(data, threshold = 0.8):
    '''Remove columns which are highly correlated.
       
       WARNING: the returned list idx_removed MUST be used in a for loop on the test data, removing features one by one
       
       Return data, idx_removed
    '''
    #initialize idx_removed
    idx_removed = []
        
    #Get first elements of the highly correlated couples
    R = np.ma.corrcoef(data.T)
    idx_HC = np.where( (R > threshold) & (R < 0.98))[0] 

    while(idx_HC.shape[0] > 0):
        
        idx_to_remove = idx_HC.max()
        
        data = np.delete(data, idx_to_remove, axis=1)
        idx_removed.append(idx_to_remove)
        
        #compute the correlation coefficients of the reduced dataset
        R = np.ma.corrcoef(data.T)
        idx_HC = np.where( (R > threshold) & (R < 0.98))[0] 
        
    
    return data, idx_removed
    

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


### Replace

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

################################################################################
#                                      PCA                                     #
################################################################################


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


################################################################################
#                               Jet division                                   #
################################################################################


def indices_jet_division(input_data):
    feature_number_of_jet = 22;
    idx0 = np.where(input_data[:,feature_number_of_jet]==0)
    idx1 = np.where(input_data[:,feature_number_of_jet]==1)
    idx2 = np.where(input_data[:,feature_number_of_jet]>=2)
    return idx0, idx1, idx2

def data_split_with_jet_division (data):
    idx0, idx1, idx2 = indices_jet_division(data)
    data_jet0 = data[idx0]
    data_jet1 = data[idx1]
    data_jet2 = data[idx2]
    return data_jet0, data_jet1, data_jet2

################################################################################
#                        Hyperparameters selection                             #
################################################################################


def grid_search_hyperparam_with_CV(y, tx, lambdas, degrees):
    loss_tr = np.zeros((len(lambdas), len(degrees)))
    loss_te = np.zeros((len(lambdas), len(degrees)))
    accuracy = np.zeros((len(lambdas), len(degrees)))
    
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):
                        
            x_augmented = build_poly(tx, degree)
            
            #regression with your favourite method
            k_indices = build_k_indices(y, 4, 1)
            acc, loss1, loss2 = cross_validation_with_ridge(y, x_augmented, k_indices, lambda_)
            
            loss_tr[idx_lambda, idx_degree] = loss1
            loss_te[idx_lambda, idx_degree] = loss2
            accuracy[idx_lambda, idx_degree] = acc
    
    #find the best using the loss
    min_loss_te = np.min(loss_te)
    best_lambda_loss = lambdas[ np.where( loss_te == min_loss_te )[0] ]
    best_degree_loss = degrees[ np.where( loss_te == min_loss_te )[1] ]

    #recompute best w
    x_augmented = build_poly(tx, int(best_degree_loss))
    best_w_loss = ridge_regression(y,x_augmented,best_lambda_loss)
    
    #find the best using the accuracy
    max_acc = np.max(accuracy)
    best_lambda_acc = lambdas[ np.where( accuracy == max_acc )[0] ]
    best_degree_acc = degrees[ np.where( accuracy == max_acc )[1] ]
    
    #recompute best w
    x_augmented = build_poly(tx, int(best_degree_acc[0]))
    best_w_acc = ridge_regression(y,x_augmented,best_lambda_acc[0])

    return best_lambda_loss, best_degree_loss, best_w_loss, best_lambda_acc, best_degree_acc, best_w_acc, loss_tr, loss_te, accuracy

def logistic_hyperparam_with_CV(y, tx, lambdas, gamma, degrees):
    
    loss_tr = np.zeros((len(lambdas), len(degrees)))
    loss_te = np.zeros((len(lambdas), len(degrees)))
    accuracy = np.zeros((len(lambdas), len(degrees)))
    
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):
                        
            x_augmented = build_poly(tx, degree)
            
            #regression with logistic method
            k_indices = build_k_indices(y, 4, 1)
            acc = cross_validation_with_logistic(y, x_augmented, k_indices, lambda_, gamma)        
            accuracy[idx_lambda, idx_degree] = acc
    
    #find the best using the accuracy
    max_acc = np.max(accuracy)
    best_lambda_acc = lambdas[ np.where( accuracy == max_acc )[0] ]
    best_degree_acc = degrees[ np.where( accuracy == max_acc )[1] ]

    return best_lambda_acc, best_degree_acc,accuracy

def grid_search_hyperparam_RIDGE(y, tx, lambdas, degrees):
    loss_tr = np.zeros((len(lambdas), len(degrees)))
    loss_te = np.zeros((len(lambdas), len(degrees)))
    
    seed = 1
    
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):
            
            x_augmented = build_poly(tx, degree)
            
            #regression with your favourite method
            x_tr, x_te, y_tr, y_te = split_data(x_augmented, y, 0.7, seed = seed)

            weights = ridge_regression(y_tr, x_tr, lambda_)

            rmse_tr= np.sqrt(2 * compute_loss_MSE(y_tr, x_tr, weights))
            rmse_vt= np.sqrt(2 * compute_loss_MSE(y_te, x_te, weights))
            loss_tr[idx_lambda, idx_degree] = rmse_tr
            loss_te[idx_lambda, idx_degree] = rmse_vt
        
    min_loss_te = np.min(loss_te)
    best_lambda = lambdas[ np.where( loss_te == min_loss_te )[0] ]
    best_degree = degrees[ np.where( loss_te == min_loss_te )[1] ]

    #recompute best w
    x_augmented = build_poly(tx, int(best_degree))
    best_w = ridge_regression(y,x_augmented,best_lambda)
    
    #version 2.0 that is easier to understand/read
    hyperparameters = [best_lambda, best_degree]
    losses = [loss_tr, loss_te]
    return hyperparameters, best_w, losses

################################################################################
#                                Logistic                                      #
################################################################################


def predict_logistic_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred=sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

sig=np.vectorize(sigmoid)

            
def convert_0_to_minus1(data):
    data[data == 0]= -1
    return data

def convert_minus1_to_0(data):
    data[data == -1]= 0
    return data

    