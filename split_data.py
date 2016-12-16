#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 12:20:26 2016

@author: user
"""

from Utils.GeneralUtils import *
list_of_top=[]
list_of_data=initialize_list_of_lists(11)


for index in range(0, 11):
    filepath=create_filename_current_directory("Datasets/all_categories_"+str(index))
    print "Working on: "+filepath
    f = open(filepath, 'r')
    lines=f.readlines()
    f.close()
    list_of_current=[]
    for line in lines:
        list_of_current.append(int(line))
    list_of_top.append(list_of_current)
    
filepath=create_filename_current_directory("Datasets/trainDmoz.txt")
X_train, y_train = load_svmlight_file(filepath)

index=0
for y in y_train:
    top_index=0
    for top in list_of_top:
        if y in top:
           list_of_data[top_index].append(X_train[index])
           list_of_data[top_index].append(y)
           