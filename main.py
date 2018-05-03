import builder
from lsi import LSI
from sklearn.datasets import fetch_20newsgroups
import sqlite3
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import h5py as h
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import neighbors as NN
from GALSF import GALSF
from GALSF_fixed import GALSF_fixed
import datasets
from sklearn.svm import SVC
import random
from sklearn.naive_bayes import GaussianNB
import time

#building datasets, includes getting the data,
#writing it to db, building term-document matrix and perforning tf-idf

#datasets.build_20newsgroups()
#datasets.build_spam()

f = h.File('hdf5/tdmatrix.hdf5', 'r')
matrix = f['tdmatrix'][:]
f.close()

#feature selection, DO NOT UNCOMMENT
##target = np.asarray(target)
##matrix = matrix.transpose()
##matrix = SelectKBest(chi2, k=1000).fit_transform(matrix, target)
##f = h.File('hdf5/tdmatrix.hdf5', 'w')
##dset = f.create_dataset('tdmatrix', data=matrix.transpose())
##f.close()

train_size = int(matrix.shape[1] * 0.6)
random.seed(10)
indices = random.sample(range(matrix.shape[1]), train_size) #indices for training set

k = 300
start = time.time()

#using GALSF
gaLSF = GALSF_fixed(k, indices, build_matrices=True, training_only=False)
X_train, X_test, y_train, y_test = gaLSF.get_best_dimensions()

#normal LSI
#X_train, X_test, y_train, y_test = LSI(k, indices, training_only=False)

end = time.time()

print("Time: " + str(end - start))

print('measuring performance')
clf = SVC(kernel='linear').fit(X_train.transpose(), y_train)
score = clf.score(X_test.transpose(), y_test)
print(score)

###clf = NN.KNeighborsClassifier(5).fit(X_train.transpose(), y_train)
###clf = GaussianNB().fit(X_train.transpose(), y_train)
###score = clf.score(X_test.transpose(), y_test)


