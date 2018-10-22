from compute_losses import *
from compute_gradients import *
from regressors import *

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
def cross_validation_with_ridge(y, x, k_indices, lambda_):
    """CV regression according to the splitting in train/test given by k_indices.
    
    The returned quantities are the average of the quantities computed in the single folds
    
    return w, loss_tr, loss_te"""
    
    folds = k_indices.shape[0]

    w_avg = np.zeros(x.shape[1])
    loss_tr_avg = 0
    loss_te_avg = 0
    
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
        loss_tr = compute_loss_RMSE(ytr,xtr,w)
        loss_te = compute_loss_RMSE(yte,xte,w)
        
        #update the quantities
        w_avg = w_avg + w/folds
        loss_tr_avg = loss_tr_avg + loss_tr/folds
        loss_te_avg = loss_te_avg + loss_te/folds
    
    return w_avg, loss_tr_avg, loss_tr_avg