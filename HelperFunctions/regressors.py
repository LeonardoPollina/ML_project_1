#Small toolbox to use in the project, this is more flexible w.r.t the 6 functions that
#we need to deliver. We can implement here our regressors

import numpy as np
from compute_losses import *
from compute_gradients import *


def grid_search(y, tx, w0, w1, compute_loss=compute_loss_MSE):
    """Algorithm for grid search.
    
    return losses"""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i,j] = compute_loss( y, tx, np.array([ w0[i],w1[j] ]) )        
    return losses


def least_squares(y, tx):
    """calculate the least squares solution.
    
    return w, lossMSE"""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    e = y - tx@w
    return w , np.mean(e**2)/2


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    return w"""
    w = np.linalg.solve(tx.T@tx + lambda_*2*y.shape[0]*np.eye(tx.shape[1]), tx.T@y)
    return w   


def gradient_descent(y, tx, initial_w, max_iters, gamma, threshold = 1e-6, 
                compute_loss=compute_loss_MSE, compute_gradient=compute_gradient_MSE):
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

    return w, loss


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


def stochastic_gradient_descent( y, tx, initial_w, batch_size, max_iters, gamma,
         compute_loss=compute_loss_MSE, compute_gradient=compute_gradient_MSE):
    """SDG algorithm

    return ws, losses"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        batch = next(batch_iter(y, tx, 32))
        minibatch_y, minibatch_tx = batch[0], batch[1]
        grad = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
        ws.append(w)
        #for the loss I use the whole dataset
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses