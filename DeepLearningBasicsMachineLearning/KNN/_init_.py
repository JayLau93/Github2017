'''
Created on 2016-7-30

@author: jieliuecnu
'''
import math
def ComputeEuclideanDistance(x1,y1,x2,y2):
    d = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return d
d_ag1 = ComputeEuclideanDistance(3,104,18,90)
d_ag2 = ComputeEuclideanDistance(2,100,18,90)
d_ag3 = ComputeEuclideanDistance(101,10,18,90)
d_ag4 = ComputeEuclideanDistance(99,5,18,90)
d_ag5 = ComputeEuclideanDistance(98,2,18,90)
d_ag6 = ComputeEuclideanDistance(98,2,18,90)
print d_ag1
print d_ag2    
print d_ag3
print d_ag4
print d_ag5
print d_ag6