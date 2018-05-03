import sqlite3
import re
import chardet
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from numpy import *
import h5py as h
import time
import operator
from sklearn.feature_extraction.text import TfidfTransformer

size = 1000 #size of a chunk of terms that are added to the matrix

#builds term-document matrix
def build_matrix(dbname):
    print ('Building the term-document matrix')
    
    con = sqlite3.connect(dbname)
    con.execute('delete from terms')
    con.commit()
    stemmer = SnowballStemmer('english')

    #parts = ['FW', 'NN', 'NNS', 'NNP', 'NNPS']

    terms = dict()
    docs = dict()
    row = 0
    col = 0
    h_size = con.execute('select count(*) from documents').fetchone()
    print ('%s documents found' % h_size[0])
    tdmatrix = zeros((1, h_size[0]), dtype='int')
    tmp_matrix = zeros((size, h_size[0]), dtype='int')
    addcount = 0
    
    for article in con.execute('select content, label from documents'):
        content = article[0]
        docs[col] = article[1]
        #content = re.sub("[^'QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm']", ' ',  content).split()
        content = word_tokenize(content)
        #content = pos_tag(content)
        for word in content:
            if len(word) < 2:
                continue
            word = word.lower()
            if word in stopwords.words('english'):
                continue
            word = stemmer.stem(word)
            if word in terms:
                if tdmatrix.shape[0] != 1:
                    if terms[word] > (tdmatrix.shape[0] - 1):
                        tmp_matrix[terms[word] - tdmatrix.shape[0], col] += 1
                    else:
                        tdmatrix[terms[word], col] += 1
                else:
                        tmp_matrix[terms[word], col] += 1
            else:
                terms[word] = row
                if addcount < size:
                    tmp_matrix[addcount, col] += 1
                else:
                    tdmatrix = vstack((tdmatrix, tmp_matrix))
                    if tdmatrix.shape[0] == size + 1:
                        tdmatrix = delete(tdmatrix, 0, 0)
                    tmp_matrix = zeros((size, h_size[0]), dtype='int')
                    addcount = 0
                    tmp_matrix[addcount, col] += 1
                row += 1
                addcount += 1
        col += 1

    if addcount < size-1:
        tmp_matrix = tmp_matrix[0:addcount]
    tdmatrix = vstack((tdmatrix, tmp_matrix))
    l = []
    to_del = []
    for r in range(tdmatrix.shape[0]):
        if count_nonzero(tdmatrix[r]) > 1:
            l.append(r)
        else:
            to_del.append(r)
    tdmatrix = tdmatrix[l,:]

    f = h.File('hdf5/tdmatrix.hdf5', 'w')
    dset = f.create_dataset('tdmatrix', data=tdmatrix)
    f.close()
    sorted_terms = sorted(terms.items(), key=operator.itemgetter(1))
    row = 0
    for word in sorted_terms:
        if word[1] not in to_del:
            con.execute('insert into terms values (?, ?)', (word[0], row))
            row += 1
    con.commit()
    con.close()
    print('DONE')

#not used
def give_ID(dbname):
    con = sqlite3.connect(dbname)
    con.execute('update documents set col=(rowid-1)')
    con.commit()
    con.close()

def tfidf(mtname):
    f = h.File('hdf5/' + mtname + '.hdf5', 'r')
    matrix = f[mtname][:]
    f.close()
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.set_params(norm=None)
    matrix = tfidf_transformer.fit_transform(matrix)
    f2 = h.File('hdf5/' + mtname + '.hdf5', 'w')
    f2.create_dataset(mtname, data=matrix.todense(), dtype='f4')
    f2.close()
    print('tf-idf performed')
