'''
Created on 2016-8-22

@author: jieliuecnu
'''
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()