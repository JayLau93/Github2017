'''
Created on 2016-8-7

@author: jieliuecnu
'''
from sklearn import svm
x = [[2,0],[1,1],[2,3]]
y = [0,0,1]
clf = svm.SVC(kernel = 'linear')
clf.fit(x,y)
print clf
print clf.support_vectors_ #get SV
print clf.support_         #get indices of SV
print clf.n_support_#get number of SVs
print clf.predict([[2,0]])