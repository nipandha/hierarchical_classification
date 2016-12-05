#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:59:37 2016

@author: user
"""

from os.path import dirname, abspath,join

    
def create_file_current_directory(file_name):
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