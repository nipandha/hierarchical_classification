#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:49:20 2016

@author: user
"""
from Utils.GeneralUtils import *

list_of_trees=[]
list_of_roots=[]
list_of_cat=[]
#path=raw_input("Enter category file path ")
#path=create_file_current_directory("Datasets/cat_hier_DMOZ.txt)"
path=create_file_current_directory("Datasets/trial_hier.txt")
f = open(path, 'r')
ctr=0
lines=f.readlines()
for line in lines:
    
    words=line.split()
    nodes=[]
    tree_index=-1
    tree_index2=-1 

    for word in words:
        nodes.append(int(word))
    
    
    #Create two checks to see if the edge connects to different subtrees  
    index=0
    for tree in list_of_trees:
        if nodes[0] in tree:
            tree_index=index
        if nodes[1] in tree:
            tree_index2=index
        index=index+1 
         
    if ((tree_index2>-1) and (tree_index>-1)):
        #Two sperate trees have now been connected by th current egde
        #Merge the trees elements,and their file data
        #set the root to the higher node 
        print len(list_of_trees)
        print len(list_of_cat)
        elements_union = list(set().union(list_of_trees[tree_index],list_of_trees[tree_index2]))    
        cat_union = list(set().union(list_of_cat[tree_index],list_of_cat[tree_index2]))    
        
        
        tree1=list_of_trees[tree_index]
        tree2=list_of_trees[tree_index2]
        list_of_trees.remove(tree1)
        list_of_trees.remove(tree2)
        list_of_trees.append(elements_union)
        cat1=list_of_cat[tree_index]
        cat2=list_of_cat[tree_index2]
        list_of_cat.remove(cat1)
        list_of_cat.remove(cat2)
        list_of_cat.append(cat_union)
        
        
        print "Old roots: "
        print list_of_roots        
        new_root=list_of_roots[tree_index]
        list_of_roots.remove(list_of_roots[tree_index2])
        list_of_roots.remove(new_root)
        list_of_roots.append(new_root)
        print "New roots: "
        print list_of_roots
        
        print "New elements of tree: "
        print elements_union
    else:
        #Set current node from 0,1
        if (tree_index==-1) and (tree_index2>-1):
             tree_index=tree_index2
        
        #if edge not connected to any existing tree     
        if tree_index==-1:
             cat=[]
             cat.append(line)
             tree=[]
             for node in nodes:
                 tree.append(node)
             list_of_cat.append(cat)
             list_of_roots.append(nodes[0])
             list_of_trees.append(tree)
             print "Set parent "+str(nodes[0])
        else: 
             #add to existing tree
             current_tree=list_of_trees[tree_index]
             list_of_cat[tree_index].append(line)
             if nodes[0] not in current_tree:
                 old=list_of_roots[tree_index]
                 list_of_roots[tree_index]=nodes[0]
                 print "Changed "+str(old)+" to "+str(list_of_roots[tree_index])                
             print "Old tree: "
             print current_tree
             for node in nodes:
                 if node not in current_tree:
                     current_tree.append(node)
             print "New tree: "
             print current_tree         

#path=raw_input("Enter dataset path ")             

index=0
for cat in list_of_cat:
    filepath=create_file_current_directory("Datasets/hierarchy_"+str(index))
    
    write_str_list_to_file(filepath,cat)
    index=index+1  

index=0
for tree in list_of_trees:
    filepath=create_file_current_directory("Datasets/all_categories_"+str(index))
    write_list_to_file(filepath,tree)
    index=index+1                  
    
filepath=create_file_current_directory("Datasets/Root_Nodes")
write_list_to_file(filepath,list_of_roots)