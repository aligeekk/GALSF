import h5py as h
import sqlite3
import operator
import math
import numpy as np
from numpy.linalg import inv
import scipy.sparse
from scipy.sparse.linalg import svds

def LSI(dim, indices, training_only=True):
##    f = h5py.File('hdf5/tdmatrix.hdf5', 'r')
##    matrix = f['tdmatrix'][:]
##    f.close()
##    smatrix = scipy.sparse.csc_matrix(matrix)
##    
##    U, S, Vt = svds(smatrix, k=dim) 
##    
##    f = h5py.File('hdf5/matrixS.hdf5', 'w')
##    f.create_dataset('matrixS', data=np.diag(S))
##    f.close()
##    f = h5py.File('hdf5/matrixU.hdf5', 'w')
##    f.create_dataset('matrixU', data=U)
##    f.close()
##    f = h5py.File('hdf5/_matrixVt.hdf5', 'w')
##    f.create_dataset('matrixVt', data=Vt)
##    f.close()
##
##    return Vt

    f = h.File('hdf5/tdmatrix.hdf5', 'r')
    matrix = f['tdmatrix'][:]
    f.close()
    f = h.File('hdf5/labels.hdf5', 'r')
    target = f['labels'][:]
    f.close()

    if training_only == False:

        smatrix = scipy.sparse.csc_matrix(matrix)
        U, S, Vt = svds(smatrix, dim)
        print('Vt shape: ' + str(Vt.shape))
        f = h.File('hdf5/Vt.hdf5', 'w')
        dset = f.create_dataset('Vt', data=Vt)
        f.close()
                   
        X_train = Vt.transpose()[indices]
        X_test = np.delete(Vt.transpose(), [x for x in indices], axis=0)
        print("X test: " + str(X_test.shape))
        print('X train: ' + str(X_train.shape))
        y_train = target[indices]
        y_test = np.delete(target, indices)
        
        X_train = X_train.transpose()
        X_test = X_test.transpose()
                
    else:
        
        X_train = matrix.transpose()[indices]
        X_test = np.delete(matrix.transpose(), [x for x in indices], axis=0)
        print("X test: " + str(X_test.shape))
        X_test = X_test.transpose()
        y_train = target[indices]
        y_test = np.delete(target, indices)
        print('X train: ' + str(X_train.shape))
        
        smatrix = scipy.sparse.csc_matrix(X_train.transpose())
        U, S, Vt = svds(smatrix, dim)
        print('U shape: ' + str(U.shape))
        print('Vt shape: ' + str(Vt.shape))
        print('S shape: ' + str(S.shape))
        X_train = Vt
        
##        f = h.File('hdf5/Vt.hdf5', 'w')
##        dset = f.create_dataset('Vt', data=Vt)
##        f.close()
##        f = h.File('hdf5/U.hdf5', 'w')
##        dset = f.create_dataset('U', data=U)
##        f.close()
##        f = h.File('hdf5/S.hdf5', 'w')
##        dset = f.create_dataset('S', data=S)
##        f.close()
##        print('DONE')

        test_converted = np.empty([dim, X_test.shape[1]])
        S = inv(np.diag(S))
        Ut = U.transpose()
        for i in range(X_test.shape[1]):
            v = X_test[:,i]
            v = np.vstack(v)
            v2 = np.dot(S, Ut)
            v2 = np.dot(v2, v)
            test_converted[:,i] = v2[:,0]
        X_test = test_converted

    return X_train, X_test, y_train, y_test

#not used
def add_new_docs():
    f = h.File('hdf5/new_docs.hdf5', 'r')
    matrix = f['pages'][:]
    f.close()
    f = h.File('hdf5/matrixU.hdf5', 'r')
    U = f['matrixUt'][:]
    Ut = np.transpose(U)
    f.close()
    f = h.File('hdf5/matrixS.hdf5', 'r')
    S = f['matrixS'][:]
    f.close()
    S = inv(S)
    col = matrix.shape[1]
    result = zeros((S.shape[0], col))
    for i in range(col):
        v = matrix[:,i]
        v = vstack(v)
        v2 = dot(Ut, v)
        v2 = dot(S, v2)
        result[:,i] = v2[:,0]
    f = h.File('hdf5/new_docs.hdf5', 'w')
    f.create_dataset('new_docs', data=result)
    f.close()
