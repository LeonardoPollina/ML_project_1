import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

sig=np.vectorize(sigmoid)

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    print(pred)
    #CorrectedZero = 1e-7
    CorrectedZero = 1e-15
    #loss = np.ones(tx.shape[0]).dot(np.log(1+np.exp(tx.dot(w))))-y.dot(tx.dot(w))
    loss = y.T.dot(np.log(pred+CorrectedZero)) + (1 - y).T.dot(np.log(1 - pred+CorrectedZero))
    print('logistic loss: ',- loss)
    return np.squeeze(- loss)


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sig(tx.dot(w))
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