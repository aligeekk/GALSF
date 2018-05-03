import numpy as np
import scipy
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import h5py as h
import scipy.sparse
from scipy.sparse.linalg import svds
import sqlite3
from sklearn.naive_bayes import GaussianNB
from scipy import linalg
from sklearn import neighbors as NN
from sklearn.svm import SVC
from numpy.linalg import inv

class GALSF_fixed:
    
    def __init__(self, dims, indices, build_matrices = True, training_only = True):
        
        self.k = dims
        self.steps = 20
        self.populationSize = 50
        self.top = 5
        self.mutations = int(self.k/20)
        self.probability = 0.8
        self.newElements = 2
        self.results = np.empty([self.populationSize])
        self.training_only = training_only

        #np.set_printoptions(threshold=np.nan)
        
        f = h.File('hdf5/tdmatrix.hdf5', 'r')
        matrix = f['tdmatrix'][:]
        f.close()

        self.population = np.empty([2, 2])

        f = h.File('hdf5/labels.hdf5', 'r')
        self.target = f['labels'][:]
        f.close()
        
        # if SVD is performed on the whole dataset
        if (self.training_only == False):
            
            # if SVD needs to be performed, else reads the matrix from the file
            if build_matrices: 
                print('performing SVD')
                U, self.S, Vt = linalg.svd(matrix, full_matrices=True)
                print('Vt shape: ' + str(Vt.shape))
                print('U shape: ' + str(U.shape))
                print('S shape: ' + str(self.S.shape))
                f = h.File('hdf5/Vt.hdf5', 'w')
                dset = f.create_dataset('Vt', data=Vt)
                f.close()
                print('DONE')
            else:
                f = h.File('hdf5/Vt.hdf5', 'r')
                Vt = f['Vt'][:]
                f.close()

            self.X_train = Vt.transpose()[indices]
            self.X_test = np.delete(Vt.transpose(), [x for x in indices], axis=0)
            print("X test: " + str(self.X_test.shape))
            print('X train: ' + str(self.X_train.shape))
            self.y_train = self.target[indices]
            self.y_test = np.delete(self.target, indices)
            
            self.X_train = self.X_train.transpose()
            self.X_test = self.X_test.transpose()

        # if SVD is performed on the training set only    
        else:
            
            self.X_train = matrix.transpose()[indices]
            self.X_test = np.delete(matrix.transpose(), [x for x in indices], axis=0)
            print("X test: " + str(self.X_test.shape))
            self.X_test = self.X_test.transpose()
            self.y_train = self.target[indices]
            self.y_test = np.delete(self.target, indices)
            print('X train: ' + str(self.X_train.shape))
            
            if build_matrices:
                
                print('performing SVD')
                self.U, self.S, Vt = linalg.svd(self.X_train.transpose(), full_matrices=True)
                print('U shape: ' + str(self.U.shape))
                print('Vt shape: ' + str(Vt.shape))
                print('S shape: ' + str(self.S.shape))
                
                f = h.File('hdf5/Vt.hdf5', 'w')
                dset = f.create_dataset('Vt', data=Vt)
                f.close()
                self.X_train = Vt
                f = h.File('hdf5/U.hdf5', 'w')
                dset = f.create_dataset('U', data=self.U)
                f.close()
                f = h.File('hdf5/S.hdf5', 'w')
                dset = f.create_dataset('S', data=self.S)
                f.close()
                print('DONE')
            else:
                f = h.File('hdf5/Vt.hdf5', 'r')
                Vt = f['Vt'][:]
                f.close()
                self.X_train = Vt
                f = h.File('hdf5/U.hdf5', 'r')
                self.U = f['U'][:]
                f.close()
                f = h.File('hdf5/S.hdf5', 'r')
                self.S = f['S'][:]
                f.close()

        self.attributes = self.S.shape[0]
        print('dimensions: ' + str(self.attributes))

    #Getting initial population or new solutions
    def get_random_vectors(self, size):
        vectors = np.zeros((size, self.attributes))
##        for v in vectors:
##            r = random.sample(range(0, self.k), self.k - self.mutations*4)
##            for i in r:
##                v[i] = 1
##            r = random.sample(range(self.k+1, self.attributes-1), self.mutations*4)
##            for i in r:
##                v[i] = 1
        for v in vectors:
            r = random.sample(range(0, self.attributes), self.k)
            for i in r:
                v[i] = 1
        return vectors

    #the main function, returns transformed training and test sets             
    def get_best_dimensions(self):
        #initial generation
        
        print('Getting initial population')
        self.population = self.get_random_vectors(self.populationSize-1)
        ktop = np.zeros(self.attributes)
        for i in range(self.k):
            ktop[i] = 1

        self.population = np.vstack((self.population, ktop))
        print('Performing evaluation')
        c = 0
        for solution in self.population:
            self.evaluate(solution, c)
            c = c + 1
        print('The best performance so far: ' + str(np.amax(self.results)))
        
        for i in range(self.steps):
            self.get_new_generation()
            c = 0 #number of the solution to evaluate
            print('Performing evaluation')
            for solution in self.population:
                self.evaluate(solution, c)
                c = c + 1
            print('The best performance so far: ' + str(np.amax(self.results)))

        max_value = np.amax(self.results)
        max_index = np.argmax(self.results)
        
        print('The best solution is:')
        print(self.population[max_index])
        print ('with accuracy ' + str(max_value))
        f = h.File('hdf5/best_solution.hdf5', 'w')
        dset = f.create_dataset('best_solution', data=self.population[max_index])
        f.close()

        clf = SVC(kernel='linear')

        #transform the test set into the new k-dimensional space
        if self.training_only:
            print(self.X_train.shape)
            self.X_train = np.delete(self.X_train, [x for x in range(0, self.attributes) if self.population[max_index][x] == 0], axis=0)
            print(np.count_nonzero(self.population[max_index]))
            self.S = np.delete(self.S, [x for x in range(0, self.attributes) if self.population[max_index][x] == 0], axis=0)
            self.U = np.delete(self.U, [x for x in range(self.attributes, self.U.shape[0])], axis=1)
            self.U = np.delete(self.U, [x for x in range(0, self.attributes) if self.population[max_index][x] == 0], axis=1)
            print(self.X_train.shape)
            print(self.S.shape)
            print(self.U.shape)
            print(self.X_test.shape)
            test_converted = np.empty([self.k, self.X_test.shape[1]])
            S = inv(np.diag(self.S))
            Ut = self.U.transpose()
            for i in range(self.X_test.shape[1]):
                v = self.X_test[:,i]
                v = np.vstack(v)
                v2 = np.dot(S, Ut)
                v2 = np.dot(v2, v)
                test_converted[:,i] = v2[:,0]
            self.X_test = test_converted
        #else simply removing singular vectors
        else:
            self.X_train = np.delete(self.X_train, [x for x in range(0, self.attributes) if self.population[max_index][x] != 1], axis=0)
            self.X_test = np.delete(self.X_test, [x for x in range(0, self.attributes) if self.population[max_index][x] != 1], axis=0)
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    #the name explains what it does: the function preserves the top n solutions and combines them
    def get_new_generation(self):
        print('Getting new generation')
        ind = np.argpartition(self.results, -self.top)[-self.top:]
        self.population = np.delete(self.population, [x for x in range(0, self.populationSize) if x not in ind], axis=0)
        top_res = np.empty(self.top)
        
        for i in range(self.populationSize - self.top - self.newElements):
            s1 = random.randint(0, self.top-1)
            s2 = random.randint(0, self.top-1)
            self.population = np.vstack((self.population, self.crossover(self.population[s1], self.population[s2])))
                       
        newSolutions = self.get_random_vectors(self.newElements)
        self.population = np.vstack((self.population, newSolutions))

    #get a new solution from two parent solutions      
    def crossover(self, s1, s2):
        ones = [x for x in range(0, self.attributes) if s1[x] == 1 or s2[x] == 1]
        #ones = np.where(s1 == 1 or s2 == 1)[0]
        
        pr = random.uniform(0, 1);
        
        #if mutates, selects k-m attributes from parents and m random ones
        if (pr <= self.probability): 
            r = random.sample(range(0, len(ones)), self.k - self.mutations)
            newSolution = np.zeros(self.attributes)
            for i in r:
                newSolution[ones[i]] = 1
            #zeros = [x for x in range(0, self.attributes) if newSolution[x] == 0]
            zeros = np.where(newSolution == 0)[0]
            r = random.sample(range(0, len(zeros)), self.mutations)
            for i in r:
                newSolution[zeros[i]] = 1
            if np.count_nonzero(newSolution) != self.k:
                print('WARNING: ' + str(np.count_nonzero(newSolution)))
                
        else: #otherwise, simple crossover
            r = random.sample(range(0, len(ones)), self.k)
            newSolution = np.zeros(self.attributes)
            for i in r:
                newSolution[ones[i]] = 1
            
        return newSolution

    #the fitness function
    def evaluate(self, solution, c):
        #clf = NN.KNeighborsClassifier(5)
        clf = SVC(kernel='linear')
        #clf = GaussianNB()

        mt = np.delete(self.X_train, [x for x in range(0, self.attributes) if solution[x] != 1], axis=0)
        scores = cross_val_score(clf, mt.transpose(), self.y_train, cv=5, scoring='accuracy')
        score = sum(scores) / float(len(scores))

        self.results[c] = score;


