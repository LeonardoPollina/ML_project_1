import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    print(pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w