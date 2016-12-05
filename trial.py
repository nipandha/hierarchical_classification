#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:54:41 2016

@author: user
"""

from os.path import dirname, abspath,join

def write_list_to_file(path)
    f = open(path, 'w')
    words=[]
    for node in tree:
        words.append(str(node))
    f.writelines(words)    
    f.close()