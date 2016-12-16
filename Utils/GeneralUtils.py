#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:59:37 2016

@author: user
"""

from os.path import dirname, abspath,join
import sklearn
import copy

def create_filename_current_directory(file_name):
    path=join(dirname(dirname(abspath(__file__))), file_name)
    
    return path
    
def write_list_to_file(path,generic_list):
    f = open(path, 'w')
    words=[]
    for node in generic_list:
        words.append(str(node)+"\n")
    f.writelines(words)    
    f.close()
    return
    
def write_str_list_to_file(path,str_list):
    f = open(path, 'w')
    
    f.writelines(str_list)    
    f.close()
    return
    
def return_current_directory(file_name):
    path=dirname(dirname(abspath(__file__)))
    
    return path
def scale_sparse_default(x_train):
    scaler = sklearn.preprocessing.MaxAbsScaler()
    # Fit to training features only and store the scalers
    scaler.fit(x_train)
    # Transform training features
    return x_train
    train_scaled = scaler.transform(x_train)
def initialize_list_of_lists(no_of_lists):
    top_list=[]
    for i in range(0,11):
        list=[]
        top_list.append(copy.copy(list))
    return top_list