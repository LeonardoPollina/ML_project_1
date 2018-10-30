import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from proj1_helpers import *

"""
Implementations of all the methods to run the project, including the requested methods.
"""

################################################################################
#                             Requested methods:                               #
################################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss_MSE(y, tx, w)
        grad = compute_gradient_MSE(y, tx, w)
        w = w - gamma*grad
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    """
    batch_size = 1
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        #get the mini batch
        batch = next( batch_iter(y, tx, batch_size) )
        minibatch_y, minibatch_tx = batch[0], batch[1]
        grad = compute_gradient_MSE(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
        #for the loss we use the whole dataset
        loss = compute_loss_MSE(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
    #solve the normal equations
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    e = y - tx@w
    loss = np.mean(e**2)/2
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    """
    w = np.linalg.solve(tx.T@tx + lambda_*2*y.shape[0]*np.eye(tx.shape[1]), tx.T@y.T)
    loss = compute_loss_rmse(y, tx, w)
    return w, loss  

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w)
        w -= gamma * grad
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    """
    w = initial_w
    for iter in range(max_iters):
        gradient = penalized_logistic_gradient(y, tx, w, lambda_)
        w -= gamma * gradient
    loss = penalized_logistic_loss(y, tx, w, lambda_)
    return w, loss


################################################################################
#                                Preprocessing                                 #
################################################################################

def replaceWithZero(data, invalidValue):
    """
    Replace each invalidValue (often used for -999 values) found in the data set with 0
    """
    numInvalidValues = countInvalid(data, invalidValue)
    idxCols = np.where(numInvalidValues>0)[0]
    for i in idxCols:
        idxInvalid = np.where(data[:,i] == invalidValue)
        data[idxInvalid,i] = 0
    return data

def standardize(data, mean = np.array([]), std = np.array([])):
    """
    Standardize the data set. 
    If the data is the training set, mean and std are computed and then applied.
    In the case the data is the test set, the mean and std used are those of the training set.
    """
    if mean.size == 0 and std.size == 0:
        mean_x = np.mean(data,axis=0)
        x = data - mean_x
        std_x = np.std(x,axis=0)
        if (std_x==0).any():
            #We substitute 0 with 0.0001 to avoid the zero division.
            std_x[std_x == 0] = 1e-4
        x = x / std_x
    else:
    #We use the mean and std provided. They are those of the training set.
        x = data - mean
        x = x / std
        mean_x,std_x = mean,std
    return x, mean_x, std_x

def indices_jet_division(data):
    """
    The 22nd feature "PRI_jet_num" represents the number of jets. 
    It is an integer with value of 0, 1, 2 or 3 with possible larger values that 
    have been capped at 3.
    Indices of samples with the same PRI_jet_num value are returned.
    """
    feature_PRI_jet_num = 22
    idx0 = np.where(data[:,feature_PRI_jet_num]==0)
    idx1 = np.where(data[:,feature_PRI_jet_num]==1)
    idx2 = np.where(data[:,feature_PRI_jet_num]>=2)
    idxs = [idx0, idx1, idx2]
    return idxs

def data_split_with_jet_division (data, idxs):
    """
    The 22nd feature "PRI_jet_num" represents the number of jets. 
    The data is divided in subsets according to the 3 array of indices contained
    the list called idxs, each one corresponding to a different jet number.
    """
    data_jet0 = data[idxs[0]]
    data_jet1 = data[idxs[1]]
    data_jet2 = data[idxs[2]]
    return data_jet0, data_jet1, data_jet2

def removeConstantColumns(data):
    """
    Remove columns which are constants, meaning a standard deviation equals to zero. 
    Return the new data and indices of the removed columns.
    """
    std = np.std(data, axis = 0)
    idx_removed = np.where(std==0)[0]
    if len(idx_removed >0 ):
        data = np.delete(data,idx_removed,axis=1)
    return data, idx_removed

def removeHighCorrelatedColumns(data, threshold = 0.8):
    """
    Remove columns which have a correlation coefficient higher than a given threshold. 
    Columns with a correlation coefficient higher than 0.98 are not deleted because this 
    value of correlation is usually obtained by the correlation of a feature with itself.
    Return the new data and indices of the removed columns

    WARNING: the returned list idx_removed MUST be used in a for loop on the test data. 
    Each deletion shifts columns indices. 
    """
    idx_removed = []
    #Compute correlation coefficients
    R = np.ma.corrcoef(data.T)
    #indices of the highly correlated (HC) columns
    idx_HC = np.where((R > threshold) & (R < 0.98))[0] 
    while(idx_HC.shape[0] > 0):
        idx_to_remove = idx_HC.max()
        data = np.delete(data, idx_to_remove, axis=1)
        idx_removed.append(idx_to_remove) 
        #compute the correlation coefficients of the reduced dataset
        R = np.ma.corrcoef(data.T)
        idx_HC = np.where( (R > threshold) & (R < 0.98))[0]       
    return data, idx_removed
    

################################################################################
#                             Cross validation                                 #
################################################################################

def build_k_indices(data, k_fold, seed):
    """
    Build k indices for k-fold.
    """
    num_row = data.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_with_ridge(y, x, k_indices, lambda_):
    """
    Cross validation using ridge regression computed on train and test sets defined by k_indices.
    Return loss of the train set, loss and accuracy of the test set averaged on the k folds.
    """
    folds = k_indices.shape[0]
    loss_tr = np.zeros(folds)
    loss_te = np.zeros(folds)
    accuracy = np.zeros(folds)
    for k in range(folds):
        #split the data in a train and a test set
        idx = k_indices[k]
        yte = y[idx]
        if len(x.shape) == 1:
            xte = x[idx]
        else:
            xte = x[idx,:]   
        ytr = np.delete(y,idx,0)
        xtr = np.delete(x,idx,0)
        #Ridge regression
        w, _ = ridge_regression(ytr,xtr,lambda_)
        #compute the rmse losses of the training and testing sets
        loss_tr[k] = compute_loss_rmse(ytr,xtr,w)
        loss_te[k] = compute_loss_rmse(yte,xte,w)
        #accuracy
        y_pred = predict_labels(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)   
    return np.mean(accuracy), np.mean(loss_tr), np.mean(loss_te)

def cross_validation_with_logistic(y, x, k_indices,initial_w, gamma, lambda_,max_iter):    
    """
    Cross validation using regularized logistic regression computed on train and test sets 
    defined by k_indices.
    Return accuracy of the test set averaged on the k folds.
    Notice that we chose to return only the accuracy, since we wanted to have a classification measure. 
    """
    folds = k_indices.shape[0]
    accuracy = np.zeros(folds)
    w=initial_w   
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
        #computing regularized logistic regression
        w, _ = reg_logistic_regression(ytr, xtr, lambda_, w, max_iter, gamma)          
        #computing accuracy
        y_pred = predict_logistic_labels(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)            
    return np.mean(accuracy)

def grid_search_hyperparam_with_RidgeCV(y, tx, lambdas, degrees):
    """
    Cross validation with grid search used to estimate th best hyperparameters for 
    ridge regression, that is lambda and the degree for polynomial expansion. 
    """
    accuracy = np.zeros((len(lambdas), len(degrees)))
    #We iterate on the hyperparameters to find the best combination of lambda and degree
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):  
            x_augmented = build_poly(tx, degree)
            k_indices = build_k_indices(y, 4, 1)
            #Cross validation with ridge regression
            acc, _, _ = cross_validation_with_ridge(y, x_augmented, k_indices, lambda_)
            #Corresponding accuracy saved
            accuracy[idx_lambda, idx_degree] = acc
    #Determine the best combination of hyperparameters that maximize the ACCURACY
    max_acc = np.max(accuracy)
    best_lambda_acc = lambdas[ np.where( accuracy == max_acc )[0] ]
    best_degree_acc = degrees[ np.where( accuracy == max_acc )[1] ]
    return best_lambda_acc, best_degree_acc, accuracy

def logistic_hyperparam_with_CV(y, tx, lambdas, gamma, degrees, max_iter):
    """
    Cross validation with grid search used to estimate th best hyperparameters for 
    regularized logistic regression, that is lambda and the degree for polynomial expansion. 
    """
    accuracy = np.zeros((len(lambdas), len(degrees)))
    #We iterate on the hyperparameters to find the best combination of lambda and degree
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):           
            x_augmented = build_poly(tx, degree)
            initial_w = np.ones((x_augmented.shape[1]))
            k_indices = build_k_indices(y, 4, 1)
            #Cross validation with regularized logistic regression
            acc = cross_validation_with_logistic(y, x_augmented, k_indices, initial_w, gamma, lambda_, max_iter)        
            accuracy[idx_lambda, idx_degree] = acc
    #Determine the best combination of hyperparameters that maximize the ACCURACY
    max_acc = np.max(accuracy)
    coordinates_best_parameter = np.where( accuracy == max_acc )
    best_lambda_acc = lambdas[ coordinates_best_parameter[0][0] ]
    best_degree_acc = degrees[ coordinates_best_parameter[1][0] ]
    return best_lambda_acc, best_degree_acc, max_acc

################################################################################
#                               Loss functions                                 #
################################################################################

def compute_loss_MSE(y, tx, w):
    """
    Calculate the loss using the mean square error (MSE)
    """
    e = y - tx.dot(w)
    return np.mean(e**2)/2

def calculate_mse(e):
    """
    Calculate the mean square error given the error vector e
    """
    return 1/2*np.mean(e**2)

def compute_loss_mse(y, tx, w):
    """
    Calculate the loss using the mean square error (mse)
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def calculate_mae(e):
    """
    Calculate the loss using the mean absolute error (mae)given the error vector e
    """
    return np.mean(np.abs(e))

def compute_loss_mae(y, tx, w):
    """
    Calculate the loss using the mean absolute error (mae)
    """
    e = y - tx.dot(w)
    return calculate_mae(e)

def calculate_rmse(e):
    """
    Calculate the root mean square error (RMSE) for the error vector e
    """
    return np.sqrt(np.mean(e**2))

def compute_loss_rmse(y, tx, w):
    """
    Calculate the root mean square error (RMSE) 
    """
    e = y - tx.dot(w)
    return calculate_rmse(e)

def calculate_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    """
    pred = sigmoid(tx.dot(w))
    #Addition of this term to avoid the computation of the log of zero
    CorrectedZero = 1e-15
    loss = y.T.dot(np.log(pred+CorrectedZero)) + (1 - y).T.dot(np.log(1 - pred+CorrectedZero))
    return np.squeeze(- loss)

def penalized_logistic_loss(y, tx, w, lambda_):
    """
    Compute the cost by negative log likelihood taking into account the regularization factor
    """
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss

################################################################################
#                             Gradient functions                               #
################################################################################

def compute_gradient_MSE(y, tx, w):
    """
    Calculate the gradient of the MSE loss
    """
    return (-1/y.shape[0]) * (tx.T) @ (y - tx @ w)

def compute_gradient_MAE(y, tx, w):
    """
    Calculate the gradient of the MAE loss
    """
    n = y.shape[0]
    grad = np.zeros(len(w))
    error = y - tx@w
    subgradient = np.sign(error)
    grad = - tx.T @ subgradient / n
    return grad

def calculate_logistic_gradient(y, tx, w):
    """
    Calculate the gradient of the MSE loss after transformation with sigmoid
    """
    #Sigmoid function rescales values between 0 and 1
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def penalized_logistic_gradient(y, tx, w, lambda_):
    """
    Calculate the gradient of the MSE loss after transformation with sigmoid,
    taking into account the regularization factor.
    """
    gradient = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient

################################################################################
#                               Predictions                                    #
################################################################################

def predict_logistic_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix using the sigmoid function.
    Notice that this fuction returns labels 0 and 1 whereas the submission file should contain
    labels -1 and 1. 
    """
    #regression prediction of test labels 
    y_pred = np.dot(data, weights)
    #Sigmoid function rescales predictions between 0 and 1 to obtain "probabilities"
    y_pred=sigmoid(y_pred)
    #Obtain the final class labels thresholding at 0.5
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def compute_accuracy_logistic(y, tx, weights):
    """
    Computes the accuracy of the predictions obtained with the function predict_logistic_labels
    which uses the sigmoid function and labels that are 0 or 1.
    """
    y_pred = predict_logistic_labels(weights, tx)
    accuracy = np.sum(y_pred == y) / len(y)  
    return accuracy

def convert_0_to_minus1(data):
    """
    Convert labels 0 to -1 value.
    """
    data[data == 0]= -1
    return data

def convert_minus1_to_0(data):
    """
    Convert labels -1 to 0 value.
    """
    data[data == -1]= 0
    return data


###############################################################################
#                           Newton functions                                  #
###############################################################################

def Compute_Hessian_Logistic(tx, w):
    '''
    Compute Hessian of the logistic loss. 
    '''
    pred = sigmoid(tx.dot(w)) 
    #Compute the diagonal of the matrix S
    Snn = pred * (1 - pred)
    #Fast way to compute diagMatrix * X, prevents memory errors
    SX = (tx.T * Snn).T
    H = tx.T @ SX
    return H


def logistic_newton(y, tx, w, gamma, max_iter, tol=1e-6):
    '''
    Logistic regression using Newton method. 
    Stopping criteria: norm2(gradient) < tol
    Return the weights w and the corresponding accuracy
    '''
    for i in range(max_iter):
        grad = calculate_logistic_gradient(y, tx, w)
        H = Compute_Hessian_Logistic(tx,w)
        w = w - gamma * np.linalg.solve(H, grad)
        if (i+1)%100 ==0:
            print(f'Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}')
        if np.abs(np.linalg.norm(grad)) < tol:
            print(f'Exit Newton, small gradient. Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}') 
            return w, compute_accuracy_logistic(y,tx,w)
    return w, compute_accuracy_logistic(y,tx,w)

def logistic_newton_regularized(y, tx, w, gamma, lambda_, max_iter, tol=1e-6):
    '''
    Logistic regression using the regularized Newton method. 
    Stopping criteria: norm2(gradient) < tol
    Return the weights w and the corresponding accuracy
    '''
    for i in range(max_iter):
        grad = calculate_logistic_gradient(y, tx, w)
        H = Compute_Hessian_Logistic(tx,w)
        #Penalization with the regularization factor
        grad = grad + 2*lambda_*w      
        H = H + np.eye(tx.shape[1])*lambda_        
        w = w - gamma * np.linalg.solve(H, grad)
        if (i+1)%100 ==0:
            print(f'Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}') 
        if np.abs(np.linalg.norm(grad)) < tol:
            print(f'Exit Newton, small gradient. Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}') 
            return w, compute_accuracy_logistic(y,tx,w)
    return w, compute_accuracy_logistic(y,tx,w)


def logistic_newton_regularized_stochastic(yFull, txFull, w, gamma, lambda_, max_iter, batch_size, tol=1e-6):
    '''
    Logistic regression using Stochastic Regularized Newton Method. 
    Stopping criteria: norm2(gradient) < tol
    Return the weights w and the corresponding accuracy.
    '''
    for i in range(max_iter):
        #Choose a random subset
        batch = next(batch_iter(yFull, txFull, batch_size))
        y, tx = batch[0], batch[1]
        grad = calculate_logistic_gradient(y, tx, w)
        H = Compute_Hessian_Logistic(tx,w)
        #Penalization with the regularization factor
        grad = grad + 2*lambda_*w      
        H = H + np.eye(tx.shape[1])*lambda_        
        w = w - gamma * np.linalg.solve(H, grad)
        if (i+1)%100 ==0:
            grad = calculate_logistic_gradient(yFull, txFull, w)
            print(f'Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}')  
        if np.abs(np.linalg.norm(grad)) < tol:
            grad = calculate_logistic_gradient(yFull, txFull, w)
            print(f'Exit Newton, small gradient. Iters = {i}, Norm2(gradient) = {np.linalg.norm(grad)}') 
            return w, compute_accuracy_logistic(yFull,txFull,w)
    return w, compute_accuracy_logistic(yFull,txFull,w)

def grid_search_hyperparam_newton_regularized(y, tx, lambdas, degrees, gamma):
    ''' 
    Grid search for penalization coefficient lambda and polynomial degree. 
    Lambdas and degrees should be of np.array() type.
    The regression is done via the Newton regularized method, with a train/set ratio of 75%/25%.
    Return best_lambda, best_degree, accuracy
    '''
    #Store all the accuracies
    accuracy = np.zeros((len(lambdas), len(degrees)))
    seed = 1
    max_iters = 500
    for idx_lambda, lambda_ in enumerate(lambdas):
        for idx_degree, degree in enumerate(degrees):
            print(f'Processing degree = {degree} and lambda = {lambda_}')
            #Creation of the augmented data x_augmented
            x_augmented = build_poly(tx, degree)
            #Regression using the logistic Newton method
            x_tr, x_te, y_tr, y_te = split_data(x_augmented, y, 0.75, seed = seed)
            w0 = np.zeros(x_tr.shape[1])
            weights, _ = logistic_newton_regularized(y_tr, x_tr, w0, gamma, lambda_, max_iters)
            accuracy[idx_lambda, idx_degree] = compute_accuracy_logistic(y_te,x_te,weights)
    #Determine the best combination of hyperparameters that maximize the ACCURACY
    max_acc = np.max(accuracy)
    best_lambda_acc = lambdas[ np.where(accuracy == max_acc)[0][0] ]
    best_degree_acc = degrees[ np.where(accuracy == max_acc)[1][0] ]
    return best_lambda_acc, best_degree_acc, accuracy

################################################################################
#                               Useful functions                               #
################################################################################

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Take as input two iterables (here the output desired values 'y' and the input data 'tx')
    Return an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data.   
    """
    data_size = len(y)
    #Possibility to apply random permutations of the samples
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
    #Creation of the indices corresponding to the mini batch 
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sample_data(y, x, seed, size_samples):
    """
    Create a sub sample of size size_samples after having performed permutation among samples. 
    """
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]
    
def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data 
    set dedicated to training and the rest dedicated to testing. 
    Return the training then testing sets (x_tr, x_te) and training then testing labels (y_tr, y_te).
    """
    #Set seed
    np.random.seed(seed)
    xrand = np.random.permutation(x)
    np.random.seed(seed)
    yrand = np.random.permutation(y)
    #Used to compute how many samples correspond to the desired ratio.
    limit = int(y.shape[0]*ratio)
    x_tr = xrand[:limit]
    x_te = xrand[(limit+1):]
    y_tr = yrand[:limit]
    y_te = yrand[(limit+1):]
    return x_tr, x_te, y_tr, y_te

def build_poly(data, degree):
    """
    Creation of the extended feature matrix using a polynomial basis up to the indicated degree.
    """
    poly_data = np.ones((len(data), 1))
    #Notice that we also consider the degree 0 corresponding to the first constant vector of ones.
    for deg in range(1, degree+1):
        poly_data = np.c_[poly_data, np.power(data, deg)]
    return poly_data

def countInvalid(data, invalidValue):
    '''
    Count how many 'invalidValues' are present in the columns of 'data'.
    Return an array with the same length as the number of features (D).
    '''
    D = data.shape[1]
    counter = np.array( [ (data[:,i] == invalidValue).sum() for i in range(D) ] )
    return counter

def removeInvalidColumns(data, invalidValue, threshold):
    ''' 
    Remove the columns containing more than threshold (in %) 
    of 'invalidValue' (often used for -999 values).
    Return the new data matrix without the invalid columns.
    '''
    numInvalidValues = countInvalid(data,invalidValue)
    numInvalidValues_percentage = numInvalidValues/(data.shape[0])
    idx_to_remove = np.where(numInvalidValues_percentage > threshold)
    data = np.delete(data,idx_to_remove,axis=1)
    return data

def replaceWithMean(data,invalidValue):
    '''
    Replace each invalidValue (often used for -999 values) found in the data set with 
    the mean value of the column which it belongs.
    '''
    numInvalidValues = countInvalid(data, invalidValue)
    idxCols = np.where(numInvalidValues>0)[0]
    for i in idxCols:
        idxValid = np.where( data[:,i] != invalidValue )
        idxInvalid = np.where( data[:,i] == invalidValue )
        mean = np.mean( data[idxValid,i] )
        data[idxInvalid,i] = mean
    return data

def sig(t):
    """
    Apply sigmoid function on t. 
    The If condition is included to use a stable version of the sigmoid function.
    """
    if(t<0):
        return np.exp(t)/(1 + np.exp(t))
    else:
        return 1.0 / (1 + np.exp(-t))

sigmoid=np.vectorize(sig)

def plot_grid_search(lambdas, degrees, accuracy, plot_name = 'dummyPlotName'):
    """
    Create a plot showing hyperparameters given the best accuracy thus selected
    among all the possibles combinations of degrees and lambdas.
    """
    #Selection of the best hyperparameters
    idx_lam = np.where(accuracy == np.max(accuracy))[0][0]
    idx_deg = np.where(accuracy == np.max(accuracy))[1][0]
    best_degree_acc = degrees[idx_deg]
    best_lambda_acc = lambdas[idx_lam]
    #Creation of the figure
    fig = plt.figure(figsize=(10,4))
    # Create a grid with 2 rows and 2 columns
    gs = GridSpec(2,2) 
    ax1 = fig.add_subplot(gs[:,1])
    Y = lambdas*1e3
    X = degrees
    X, Y = np.meshgrid(X, Y)
    Z = 1 - accuracy
    loss = ax1.contourf(X, Y, Z, 100)
    ax1.plot(best_degree_acc, best_lambda_acc*1e3, '8', color='r', markersize=13)
    fig.colorbar(loss, ax=ax1)
    #Labels and title of the plot
    ax1.set_ylabel('Lambdas (1e-3)')
    ax1.set_xlabel('Degrees')
    ax1.set_title('1 - accuracy')
    ax2 = fig.add_subplot(gs[0,0])
    ax2.plot(lambdas*1e3, 1-accuracy[:,idx_deg])
    ax2.set_ylabel('1 - accuracy')
    ax2.set_xlabel('Lambdas (1e-3)')
    ax2.set_title(f'Degree = {best_degree_acc}')
    starx = best_lambda_acc*1e3
    stary = np.min(1-accuracy[:,idx_deg])
    ax2.plot(starx, stary, '8', color='r', markersize=7)
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(degrees, 1-accuracy[idx_lam,:])
    ax2.set_ylabel('1 - accuracy')
    ax2.set_xlabel('Degrees')
    ax2.set_title(f'Lambda = {best_lambda_acc*1e3}*1e-3')
    starx = best_degree_acc
    stary = np.min(1-accuracy[idx_lam,:])
    ax2.plot(starx, stary, '8', color='r', markersize=7)
    fig.subplots_adjust(hspace=.7)
    fig.savefig(plot_name, format = 'eps')

################################################################################
#                                     PCA                                      #
################################################################################

def PCAWithCovariance(data):
    '''
    Compute the eigenvalues and eigenvector of the COVARIANCE matrix of data. 
    Return: percVariance (a vector containing the explained variance by each single feature) 
            eVectors (a matrix in which the columns contain ordered eigenvectors, that is the possible 
                 linear combinations of features useful to create new principal components)
    Notice that data has to be standardized. 
    '''
    #covariance matrix (COV)
    COV = np.cov(data, rowvar = False)
    #ordered eValues and eVectors of COV
    eValues, eVectors = np.linalg.eig(COV)
    #contributions of each variables to the variance
    totVariance = sum(eValues)
    percVariance = eValues/totVariance
    return percVariance, eVectors
  
def PCAWithScatterMatrix(data):
    '''
    Compute the eigenvalues and eigenvector of the SCATTER matrix of data. 
    Return: percVariance (a vector containing the explained variance by each single feature) 
            eVectors (a matrix in which the columns contain ordered eigenvectors, that is the possible 
                 linear combinations of features useful to create new principal components)
    Notice that data has to be standardized. 
    '''
    #scatter matrix (XX)
    XX = data.T @ data
    #ordered eValues/eVectors of XX
    eValues, eVectors = np.linalg.eig(XX)
    #contributions of each variables to the variance
    totVariance = sum(eValues)
    percVariance = eValues/totVariance
    return percVariance, eVectors
