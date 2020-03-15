import numpy as np
from matplotlib import pyplot as plt

from datasets import dataset_1,calculoDeErrorCross,XBad
from gradient_descent import gradient_descent
from linear_cost import linear_cost
from linear_cost_derivate import linear_cost_derivate

# Training data
(X, y) = dataset_1
m, n = X.shape

#theta_0 = np.random.rand(n, 1)
theta_0 =np.vstack([[0.0774702 ],[0.89021182],[0.97577675]])


'''
for i in range(1,100):
	theta, costs, gradient_norms = gradient_descent(
	    X,
	    y,
	    theta_0,
	    linear_cost,
	    linear_cost_derivate,
	    alpha=0.0000001, #tamaño de pasos
	    treshold=0.001,#Que tanto quiero que se apegue al 0 en el gradient descent
	    max_iter=10000,
	    Lambda=i #regularizacion, que tanto quiero que se apegue a mis datos
	)

	#linearCosts.append(linear_cost(X,y,theta,1))
	#linearCosts.append(calculoDeErrorCross(theta))
	#iteration.append(i)
	print("Lambda "+str(i)+"\t-\t"+str(calculoDeErrorCross(theta)))
'''

for i in range(3):
	theta, costs, gradient_norms = gradient_descent(
		    X,
		    y,
		    theta_0,
		    linear_cost,
		    linear_cost_derivate,
		    alpha=0.0000001, #tamaño de pasos
		    treshold=0.001,#Que tanto quiero que se apegue al 0 en el gradient descent
		    max_iter=10000,
		    Lambda=3 #regularizacion, que tanto quiero que se apegue a mis datos
		)

print ('THETA:', theta)
print(calculoDeErrorCross(theta))

# Plot training data
#puntos de datos
plt.scatter(X[:, 1], y) 

#Linea de regresion
plt.scatter(X[:, 1], np.matmul(X, theta), color='red')

#Grafica de codo ////creo
#plt.plot(np.arange(len(costs)), costs)


#Grafica de costo vs Lambda
#plt.plot(iteration, linearCosts, color='red')

plt.show()
