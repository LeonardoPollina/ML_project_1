import numpy as np

################################################################################
#                        Some helper functions                                 #
################################################################################

def compute_loss_MSE(y, tx, w):
    """Calculate the loss using MSE
    """
    e = y - tx.dot(w)
    return np.mean(e**2)/2

def compute_gradient_MSE(y, tx, w):
    return (-1/y.shape[0]) * (tx.T) @ (y - tx @ w)

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



################################################################################
#                        The requested methods:                                #
################################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss_MSE(y, tx, w)
        grad = compute_gradient_MSE(y, tx, w)
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):
        #get the mini batch
        batch = next( batch_iter(y, tx, 1) )
        minibatch_y, minibatch_tx = batch[0], batch[1]

        grad = compute_gradient_MSE(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad

        #for the loss we use the whole dataset
        loss = compute_loss_MSE(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares(y, tx):
    """Compute the least squares solution."""

    #solve the normal equations
    w = np.linalg.solve(tx.T@tx, tx.T@y)

    e = y - tx@w
    loss = np.mean(e**2)/2
    return w, loss



def ridge_regression(y, tx, lambda_):
    """Compute ridge regression."""
    w = np.linalg.solve(tx.T@tx + lambda_*2*y.shape[0]*np.eye(tx.shape[1]), tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss   









