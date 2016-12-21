import copy
import pandas as pd
import numpy as np
import Hierarchical_Classification.mock_model as mock
import Hierarchical_Classification.hierarchy as h

class HierarchyClassification:

    def __init__(self, cat, base_model, C=1.0):

        self.cat = cat
        self.base_model = copy.deepcopy(base_model)
        self.C = C
        self.base_model.C = self.C

        # Building hierarchy
        print("Building hierarchy for category: %s" % self.cat)
        self.hier = h.Hierarchy(self.cat)
        self.hier.build_levels()

        self.node_models = {}

    def predict_row(self, X_row):

        pred_y = self.hier.root

        while True:
            pred_y = self.node_models[pred_y].predict(X_row)[0]

            if (pred_y == -1):
                return pred_y

            if (self.hier.is_leaf(pred_y)):
                return self.hier.get_real_leaf(pred_y)

        return -1.0

    def predict_all(self, X):

        pred_y_vect = []

        for row_num in range(X.shape[0]):
            X_row = X[row_num]
            pred_y_vect.append(self.predict_row(X_row))

        return np.array(pred_y_vect)

    def score(self, X, y):

        y_pred = self.predict_all(X)
        mask = (y_pred != y)

        return 1.0 - (mask.sum())/len(y)

    def fit(self, scl_X_train_classif, y_train_classif):

        for level, level_nodes in self.hier.levels.items():
            print("We are on level: %s and nodes: %s" % (level, level_nodes))

            for node in level_nodes:

                if self.hier.is_leaf(node):
                    continue

                children_nodes = self.hier.get_immediate_children_node(node, keep_leaves=False)

                print("Fitting for node: %s and its children: %s" % (node, children_nodes))

                # Slice the data
                level_X_train, level_y_train = self.hier.slice_data(scl_X_train_classif, y_train_classif, children_nodes)
                # Transform y_labels
                level_y_train = self.hier.change_y(level_y_train, children_nodes)
                # Fit model
                try:
                    model_use = copy.deepcopy(self.base_model)
                    model_use.fit(level_X_train, level_y_train)
                    self.node_models[node] = model_use
                except:
                    print("Couldn't fit")
                    # This happens because there is only one distinct value in the y array
                    # or the y array is empty. This label does not exist in "train for classification", but exists
                    # in "train for hierarchy"

                    if len(level_y_train > 0):
                        mock_model = mock.MockModel(level_y_train[0])
                    else:
                        mock_model = mock.MockModel(-1)

                    self.node_models[node] = mock_model

        # print("Pickling the node models!")
        # pd.to_pickle(self.node_models, h.ROOT+("Fitted_Hierarchy//%s_Fitted_HC.p" % self.cat))



