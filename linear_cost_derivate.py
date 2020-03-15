import numpy as np


def linear_cost_derivate(X, y, theta,Lambda=1):
    h = np.matmul(X, theta)
    m, _ = X.shape
    thetaJ=h
    newLanbda=np.empty(m)
    newLanbda.fill(Lambda)
    return (np.matmul((h - y).T, X).T+np.matmul(newLanbda,thetaJ).T) / m