import numpy as np


def linear_cost(X, y, theta,Lambda=1):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    thetaJ=h**2
    return (sq.sum()+(Lambda*thetaJ.sum())) / (2 * m)