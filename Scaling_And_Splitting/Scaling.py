import Flat_Classification.constants as ct
import pandas as pd
import numpy as np
import numpy.random as np_random
import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.utils

def scale_and_pickle():

    for cat in range(0, 11):
        print("Scaling and outputting category: %s" % cat)

        #  Getting the data
        X_train_hier = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_X_train_hier.p" % cat)
        y_train_hier = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_y_train_hier.p" % cat)

        assert (X_train_hier.shape[0] == len(y_train_hier))

        X_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_X_train_classif.p" % cat)
        y_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_y_train_classif.p" % cat)

        assert (X_train_classif.shape[0] == len(y_train_classif))

        X_test = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_X_test.p" % cat)
        y_test = pd.read_pickle(ct.ROOT + "Pickles\\Unscaled\\%s_y_test.p" % cat)

        assert (X_test.shape[0] == len(y_test))

        # Scaling
        # Scale train hier and train classif independently
        scaler_train_hier = sklearn.preprocessing.MaxAbsScaler()
        scaler_train_classif = sklearn.preprocessing.MaxAbsScaler()

        scl_X_train_hier = scaler_train_hier.fit_transform(X_train_hier)
        scl_X_train_classif = scaler_train_classif.fit_transform(X_train_classif)
        # Use train classif scalers for test sample
        scl_X_test = scaler_train_classif.transform(X_test)

        # Outputting to pickles
        pd.to_pickle(scl_X_train_hier, ct.ROOT + "Pickles\\Scaled\\%s_X_scl_train_hier.p" % cat)
        pd.to_pickle(scl_X_train_classif, ct.ROOT + "Pickles\\Scaled\\%s_X_scl_train_classif.p" % cat)
        pd.to_pickle(scl_X_test, ct.ROOT + "Pickles\\Scaled\\%s_X_scl_test_hier.p" % cat)

        # Outputting to liblinear/libsvm format
        sklearn.datasets.dump_svmlight_file(scl_X_train_hier, y_train_hier,
                                            f=ct.ROOT + "Text\\Scaled\\%s_scl_train_hier.txt" % cat)
        sklearn.datasets.dump_svmlight_file(scl_X_train_classif, y_train_classif,
                                            f=ct.ROOT + "Text\\Scaled\\%s_scl_train_classif.txt" % cat)
        sklearn.datasets.dump_svmlight_file(scl_X_test, y_test,
                                            f=ct.ROOT + "Text\\Scaled\\%s_scl_test.txt" % cat)

        return None