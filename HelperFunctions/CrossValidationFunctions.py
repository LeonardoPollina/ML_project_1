from compute_losses import *
from compute_gradients import *
from regressors import *
from proj1_helpers import *

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
        loss_tr[k] = compute_loss_RMSE(ytr,xtr,w)
        loss_te[k] = compute_loss_RMSE(yte,xte,w)

        #accuracy
        y_pred = predict_labels(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)  

    if printSTD:
        print(f'STD of test error: {np.std(loss_te)}')

    
    return np.mean(accuracy), np.mean(loss_tr), np.mean(loss_te)