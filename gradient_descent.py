import numpy as np


def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        alpha=0.01,
        treshold=0.0001,
        max_iter=20000,
        Lambda=1):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta,Lambda)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta,Lambda)
        i += 1
        costs.append(cost(X, y, theta,Lambda))
        gradient_norms.append(cost_derivate(X, y, theta,Lambda))
        #print("Iteracion",str(i))
    return theta, costs, gradient_norms
