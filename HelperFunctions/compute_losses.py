import numpy as np

def calculate_mse(e):
    """Calculate the MSE for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the MAE for vector e."""
    return np.mean(np.abs(e))


def calculate_rmse(e):
    """Calculate the RMSE for vector e."""
    return np.sqrt(np.mean(e**2))


def compute_loss_MSE(y, tx, w):
    """Calculate the MSE loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_MAE(y, tx, w):
    """Calculate the MAE loss. """

    e = y - tx.dot(w)
    return calculate_mae(e)

def compute_loss_RMSE(y, tx, w):
    """Calculate the RMSE loss. """

    e = y - tx.dot(w)
    return calculate_rmse(e)