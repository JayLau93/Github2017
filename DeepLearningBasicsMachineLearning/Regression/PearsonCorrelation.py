'''
Created on 2016-9-2

@author: jieliuecnu
'''
import numpy as np
import math
from astropy.units import Ybarn
 
def computerCorrelation(X,Y):
     xBar = np.mean(X)
     yBar = np.mean(Y)
     SSR = 0
     varX = 0
     varY = 0
     for i in range(0,len(X)):
         diffXXBar = X[i] - xBar
         diffYYBar = Y[i] - yBar
         SSR += (diffXXBar*diffYYBar) 
         varX += diffXXBar**2
         varY += diffYYBar**2
     SST = math.sqrt(varX*varY)
     return SSR/SST
def polyFit(X,Y,degree):
    results = {}
    coeffs = np.polyfit(X,Y,degree)
    results["polynomial"] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(X)
    ybar = np.mean(Y)
    ssreg = np.sum((yhat-ybar)**2)
    print "ssreg:",ssreg
    sstot = np.sum((Y-ybar)**2)
    print "sstot:",ssreg
    results["determination"] = ssreg/sstot
    return results
testX = [1,3,8,7,10]
testY = [10,12,24,21,34]
r = computerCorrelation(testX,testY)
print "r:",  r
print "r^2:", r**2
print "r^2:", polyFit(testX,testY,1)    
