import pandas as pd
import numpy as np

import Flat_Classification.constants as ct

def get_avg_features_cat(cat):

    # Reading X pickles
    scl_X_train_hier = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_X_scl_train_hier.p" % cat)

    # Reading y pickles
    y_train_hier = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_train_hier.p" % cat)

    return get_avg_features(scl_X_train_hier, y_train_hier)

def remove_zero_columns(a):

    print("Dropping non-zero columns!")

    ncols = a.shape[1]
    non_zero_cols = []

    sum_cols = a.sum(axis=0)

    for i in range(ncols):
        if np.abs(sum_cols[:, i]) > 1e-5:
            non_zero_cols.append(i)

    return a[:, non_zero_cols]

def _get_avg_feature(X, mask):

    return X[mask].sum(axis=0) / mask.sum()

def get_avg_features(X, y):

    X_use = remove_zero_columns(X)
    del X

    mapping = {}

    y_unique = np.unique(y)

    avg_feature_rows = []

    for i, y_value in enumerate(y_unique):
        mapping[i] = y_value
        mask = (y == y_value)
        avg_feature = _get_avg_feature(X_use, mask)
        avg_feature_rows.append(avg_feature)
        del avg_feature
        del mask

    avg_feature_matrix = np.concatenate(avg_feature_rows, axis=0)

    return avg_feature_matrix, mapping

if __name__ == '__main__':

   for cat in range(10):

        print("Building average feature matrix for category: %s" % cat)
        avg_feat_matrix, mapping = get_avg_features_cat(cat)
        shape = avg_feat_matrix.shape
        print("Average feature matrix has has shape %s, %s. Now pickling!" % (shape[0], shape[1]))

        pd.to_pickle(avg_feat_matrix, ct.ROOT + "Pickles\\Avg_Feature_Matrices\\%s_Avg_Feat_Matrix.p" % cat)
        pd.Series(mapping).to_csv(ct.ROOT + "Pickles\\Avg_Feature_Matrices\\%s_Feat_Matrix_Mapping.csv" % cat)

        del avg_feat_matrix