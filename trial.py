#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:54:41 2016

@author: user
"""
from Utils.GeneralUtils import *
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import pandas as pd
filepath=create_filename_current_directory("Datasets/trainDmoz1.txt")
x_train, y_train = load_svmlight_file(filepath)

scaler = preprocessing.MaxAbsScaler()
# Fit to training features only and store the scalers
scaler.fit(x_train)
# Transform training features
train_scaled = scaler.transform(x_train)

print train_scaled

'''from os.path import dirname, abspath,join

def write_list_to_file(path)
    f = open(path, 'w')
    words=[]
    for node in tree:
        words.append(str(node))
    f.writelines(words)    
    f.close()'''