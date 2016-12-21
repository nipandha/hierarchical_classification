import sklearn
import sklearn.svm
import pandas as pd
import numpy as np
import copy

ROOT = r"C:\Users\Vlad Dobru\Desktop\DataFinal\Pickles\\"

class Hierarchy:
    def __init__(self, cat, type='Cos_Complete_2'):

        self.cat = cat
        self.type = type
        self.children_rel = pd.read_csv(
            ROOT + "Hierarchies_Final\\Hierarchy\\%s\\%s_children.csv" % (self.type, cat), header=None)

        self.mappings_to_real = pd.read_csv(
            ROOT + "Hierarchies_Final\\Mapping_Leaves_To_Real_Classes\\%s_Feat_Matrix_Mapping.csv" % cat, header=None)[
            1]

        self.n_leaves = len(self.mappings_to_real)

        self.children_rel.index = range(self.n_leaves, self.n_leaves + len(self.children_rel))
        self.root = self.children_rel.index[-1]

        self.children_rel_dict = self.children_rel.T.to_dict("list")

        self.levels = {}

    def get_leaves(self):

        return list(range(0, self.n_leaves))

    def is_leaf(self, node):

        return (node < self.n_leaves)

    def get_real_leaf(self, leaf):

        assert (self.is_leaf(leaf))

        return self.mappings_to_real.ix[leaf]

    def get_parent(self, node):

        if node==self.root:
            return node

        for parent, immediate_children in self.children_rel_dict.items():
            if (node in immediate_children):
                return parent

        return None

    def get_immediate_children_node(self, node, keep_leaves=False):

        if (keep_leaves):
            fallback = [node]
        else:
            fallback = []

        return self.children_rel_dict.get(node, fallback)

    def get_path_from_root(self, node):

        if node==self.root:
            return [node]

        path = [node]
        parent = node

        while True:
            parent = self.get_parent(parent)
            path.append(parent)

            if parent==self.root:
                break

        return sorted(path, reverse=True)

    def get_immediate_children_nodes(self, nodes, keep_leaves=False):

        children = []

        for node in nodes:
            children += self.get_immediate_children_node(node, keep_leaves)

        return children

    def get_ultimate_leaves(self, node):

        leaves = [node]

        while (True):
            leaves = self.get_immediate_children_nodes(leaves, keep_leaves=True)
            if self.all_leaves(leaves):
                break

        return leaves

    def get_ultimate_real_leaves(self, node):

        hier_leaves = self.get_ultimate_leaves(node)

        return [self.get_real_leaf(leaf) for leaf in hier_leaves]

    def get_ultimate_real_leaves_nodes(self, nodes):

        all_real_leaves = []

        for node in nodes:
            all_real_leaves += self.get_ultimate_real_leaves(node)

        return all_real_leaves

    def slice_data(self, X, y, nodes):

        real_leaves = self.get_ultimate_real_leaves_nodes(nodes)
        y_series = pd.Series(y)
        mask = y_series.isin(real_leaves).values
        y_slice = y_series.ix[mask].values

        return X[mask], y_slice

    def change_y(self, y, nodes):

        y_new = copy.deepcopy(y)

        for node in nodes:
            leaves_for_node = self.get_ultimate_real_leaves(node)
            for i in range(0, len(y_new)):
                if (y_new[i] in leaves_for_node):
                    y_new[i] = node

        return y_new

    def all_leaves(self, nodes):

        for node in nodes:
            if ~(self.is_leaf(node)):
                return False

        return True

    def build_levels(self):

        count = 0
        self.levels[count] = [self.root - count]

        while (True):
            count += 1
            print(count - 1, self.levels[count - 1])

            this_level = self.get_immediate_children_nodes(self.levels[count - 1])

            if (len(this_level) == 0):
                break
            else:
                self.levels[count] = this_level