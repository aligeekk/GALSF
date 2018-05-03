The program classifies documents using SVM after performing LSI or GALSF. The following functions are available:

To obtain the datasets:

datasets.build_20newsgroups() - downloads the 20 Newsgroups dataset, saves it to a sqlite3 database, builds the term-document matrix, performs tf-idf and saves the result to an hdf5 file
datasets.build_spam() - extracts the SMS spam dataset from the .csv file, other steps are the same

To perform GALSFf:

gaLSF = GALSF_fixed(k, indices, build_matrices=True, training_only=False)
where
k - the number of dimensions
indices - indices of instances in the training set (to keep them the same in all the experiments)
build_matrices - whether the matrices should be built or just loaded from an hdf5 file if built before
training_only - whether SVD is performed on the training set only and the test set is added later
The term-document matrix is taken from the hdf5 file and don't need to be passed as a parameter.

X_train, X_test, y_train, y_test = gaLSF.get_best_dimensions() - returns the transformed test and training sets

To perform normal LSI:

X_train, X_test, y_train, y_test = LSI(k, indices, training_only=False)
parameters and the output are the same

After these steps, classification is performed using linear SVM from scikit-learn.