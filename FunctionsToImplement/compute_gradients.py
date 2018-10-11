import numpy as np

def compute_gradient_MSE(y, tx, w):
    return (-1/y.shape[0]) * (tx.T) @ (y - tx @ w)



def compute_gradient_MAE(y, tx, w):
    n = y.shape[0]
    grad = np.zeros(len(w))
    error = y - tx@w
    subgradient = np.sign(error)
    grad = - tx.T @ subgradient / n
    return grad