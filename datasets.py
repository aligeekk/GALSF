import builder
from sklearn.datasets import fetch_20newsgroups
import sqlite3
import h5py as h
import numpy as np
import csv
import os

def build_20newsgroups():

    categories = [ 'sci.med', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    try:
        os.remove('20newsgroups.db')
    except OSError:
        pass
    
    con = sqlite3.connect('20newsgroups.db')
    
    con.execute('CREATE TABLE terms (term TEXT, row NUMERIC)')
    con.execute('CREATE TABLE documents (col INTEGER PRIMARY KEY, content TEXT, label NUMERIC)')
    con.commit()

    for i in range(len(newsgroups.target)):
        con.execute('insert into documents(content, label) values (?,?)', (newsgroups.data[i], int(newsgroups.target[i])))

    con.commit()
    con.close()

    f = h.File('hdf5/labels.hdf5', 'w')
    dset = f.create_dataset('labels', data=list(map(int, newsgroups.target)))
    f.close()

    builder.build_matrix('20newsgroups.db')
    builder.tfidf('tdmatrix')
    

def build_spam():

    try:
        os.remove('spam.db')
    except OSError:
        pass
    
    con = sqlite3.connect('spam.db')
    
    con.execute('CREATE TABLE terms (term TEXT, row NUMERIC)')
    con.execute('CREATE TABLE documents (col INTEGER PRIMARY KEY, content TEXT, label NUMERIC)')
    con.commit()

    spam = []
    ham = []

    with open('spam.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0] == 'ham':
                ham.append(row[1])
            else:
                spam.append(row[1])

    ham = ham[0:len(spam)]

    print(len(ham))
    print(len(spam))

    labels = [0] * len(ham) + [1] * len(spam)

    for i in range(len(ham)):
        con.execute('insert into documents(content, label) values (?,?)', (ham[i], int(0)))

    for i in range(len(spam)):
        con.execute('insert into documents(content, label) values (?,?)', (spam[i], int(1)))

    con.commit()
    con.close()

    f = h.File('hdf5/labels.hdf5', 'w')
    dset = f.create_dataset('labels', data=list(labels))
    f.close()

    builder.build_matrix('spam.db')
    builder.tfidf('tdmatrix')
    

def build_wiki():

    con = sqlite3.connect('wiki.db')
    con.execute('CREATE TABLE terms (term TEXT, row NUMERIC)')
    con.execute('CREATE TABLE documents (col INTEGER PRIMARY KEY, content TEXT, label NUMERIC)')
    con.commit()

    #for i in range(len(newsgroups.target)):
        #con.execute('insert into documents(content, label) values (?,?)', (newsgroups.data[i], int(newsgroups.target[i])))

    con.commit()
    con.close()

    builder.build_matrix('wiki.db')
    builder.tfidf('tdmatrix')
