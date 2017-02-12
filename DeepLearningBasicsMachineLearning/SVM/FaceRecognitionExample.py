'''
Created on 2016-8-9

@author: jieliuecnu
'''
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


print(__doc__)
#display progress 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
#download data
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
#introspect the images arrays to find the shapes(for plotting)
#print(lfw_people.image.shape)
n_samples, h, w = lfw_people.image.shape
#positions info is ignored by this model
X = lfw_people.data
n_features = X.shape[1]
#the label to predict is the id of the person
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size: ")
print("n_samples: %d"%n_samples)
print("n_features: %d"%n_features)
print("n_classes: %d"%n_classes)
#split into a training and testing set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
#compute a pca on the face dataset(treated as unlabeled)
n_components = 150
print("Extracting the top %d eigenfaces from %d faces"
      %(n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X.train)
print("done in %0.3fs"%(time() - t0))
eigenface = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.trnsform(X_test)
print("done in %0.3fs"%(time() - t0))
#train a svm classification Model
print("fitting the classifier to the training set")
t0 = time()
param_grid = {'c':[1e3,5e3,1e4,5e4,1e5],
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],}
clf = GridSearchCV(SVC(kernel='rgf', class_weight='auto'),param_grid)
clf = clf.fit(X_train_pca,Y_train)
print("done in %0.3fs"%(time() - t0))
print("best estimator found by grid search:")
print(clf.best_estimator_)
#quantitative evaluation of the model quality on the test set
print("predicting people's names on the test set")
t0 = time()
Y_pred = clf.predict(X_test_pca)
print("done in %0.3fs"%(time() - t0))
print(classification_report(Y_test,Y_pred,target_names=target_names))
print(confusion_matrix(Y_test, Y_pred, labels=range(n_classes)))
#quantitative evaluation of the model quality on the test set
def plot_gallery(images,titles,h,n_row=3,n_col=4):
    "helper function to plot a gallery of portraits"
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks()
        plt.yticks()
def title(Y_pred, Y_test, target_names, i):
    pred_name = target_names[Y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[Y_test[i]].rsplit(' ',1)[-1]
    return 'predicted: %s\n true:     %s'%(pred_name,true_name)
prediction_titles = [title(Y_pred,Y_test,target_names,i)
                     for i in range(Y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)
#plot the gallery of the most significant eigenfaces
eigenface_titles = ["eigenface %d"% i for i in range(eigenface.shape[0])]
plot_gallery(eigenface, eigenface_titles, h, w)
plt.show()
