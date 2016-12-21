import pandas as pd
import array
from os.path import dirname, abspath,join

ROOT = r"C:\Users\Vlad Dobru\Desktop\DataFinal\Pickles\\"

def write_list_to_file(path,generic_list):
    f = open(path, 'w')
    words=[]
    for node in generic_list:
        words.append(str(node)+"\n")
    f.writelines(words)
    f.close()
    return

def create_filename_current_directory(file_name):
    path = join(ROOT, file_name)

    return path

def write_list_of_list_to_file(path,generic_lists,alpha=0):
    f = open(path, 'w')
    if(alpha!=0):
        f.writelines(str(alpha)+"\n")

    for generic_list in generic_lists:
        words=[]
        for node in generic_list:
            words.append(str(node)+"\n")
        words.append("-999\n")
        f.writelines(words)
    f.close()
    return

def run(cat):
    df=pd.read_csv(ROOT + (r"Hierarchies_Final/Hierarchy/Cos_Average_2/%s_children.csv" % cat),header=None)
    df1=pd.read_csv(ROOT + ("Hierarchies_Final/Mapping_Leaves_To_Real_Classes/%s_Feat_Matrix_Mapping.csv") % cat,
                    header=None)

    all_nodes=list(df1[0])
    parent_mappings = [0 for i in range(201)]

    left=list(df[0])
    right=list(df[1])
    cnt=len(all_nodes)
    # leaf_cnt=cnt
    for l,r in zip(left,right):
        all_nodes.append(cnt)

        parent_mappings[int(l)]=cnt
        parent_mappings[int(r)]=cnt
        cnt = cnt + 1
    total_cnt=cnt
    top_nodes=[]
    levels=[]


    cnt=0
    visited=len(all_nodes)
    for i in range(0,visited):
        if (parent_mappings[i]==0):
            top_nodes.append(cnt)
            visited=visited-1
        cnt = cnt + 1
    levels.append(top_nodes)
    current_top=0
    while(visited>0):
        current_level=[]
        for i in range(0,total_cnt):
            if parent_mappings[i] in levels[current_top]:
                current_level.append(i)
                visited -= 1
        levels.append(list(current_level))
        current_top=current_top+1

    path=create_filename_current_directory(r"Hierarchies_Final/Level_0")
    write_list_of_list_to_file(path,levels,len(all_nodes))
    path=create_filename_current_directory(r"Hierarchies_Final/Parent_0")
    write_list_to_file(path,parent_mappings[:len(all_nodes)])
    i=0
    for level in levels:
        print("The number of nodes at level %d is %d"%(i,len(level)))
        i=i+1

    return len(all_nodes), levels, parent_mappings[:len(all_nodes)]