'''
Created on 2016-8-31

@author: jieliuecnu
'''
import numpy as np
import random
import matplotlib.pyplot as plt
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0,numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y 
        cost = np.sum(loss**2)/(2*m)
        print("Iteration: %d,Cost: %f"%(i,cost))
        gradient = np.dot(xTrans, loss)/m
        theta = theta - alpha*gradient
    return theta    
def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=numPoints)
    for i in range(0,numPoints):
        x[i][0] = 1;
        x[i][1] = i;
        y[i] = (i + bias) + random.uniform(0,1)*variance
    return x,y
x,y = genData(100,25,10)
# print "x:"
# print x
# print "y:"
# print y
m,n = np.shape(x)
n_y = np.shape(y)
print "x shape:", m, " ", n
print "y length:", n_y
# plt.plot(x[:,0],y,"--",color="red",marker="<")
# plt.plot(x[:,1],y,"-",color="green",marker="o")
# plt.show()
numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)