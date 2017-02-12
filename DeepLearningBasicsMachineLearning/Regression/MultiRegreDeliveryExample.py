'''
Created on 2016-8-27

@author: jieliuecnu
'''
from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = "D:\\Delivery.csv"
deliveryData = genfromtxt(dataPath, delimiter=",")
print "data"
print deliveryData
X = deliveryData[:,:-1]
Y = deliveryData[:,-1]
print "X:", X
print "Y:", Y
regr = linear_model.LinearRegression()
regr.fit(X,Y)
print "coefficients"
print regr.coef_
print "intercept: "
print regr.intercept_
xPred = [102,6]
yPred = regr.predict(xPred)
print "predicted y:"
print yPred