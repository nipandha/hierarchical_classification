import pandas as pd

import Flat_Classification.constants as ct
import Hierarchical_Classification.hc_model as hcm
import Hierarchical_Classification.hierarchy as h

if __name__ == '__main__':

    cat = 0

    # Reading X pickles
    scl_X_train_classif = pd.read_pickle(h.ROOT + "Scaled\\%s_X_scl_train_classif.p" % cat)
    scl_X_test = pd.read_pickle(h.ROOT + "Scaled\\%s_X_scl_test_hier.p" % cat)

    # Reading y pickles
    y_train_classif = pd.read_pickle(h.ROOT + "Scaled\\%s_y_train_classif.p" % cat)
    y_test = pd.read_pickle(h.ROOT + "Scaled\\%s_y_test.p" % cat)

    hc_model = hcm.HierarchyClassification(cat, ct.MODELS["LR_OVR"], C=0.1)
    hc_model.fit(scl_X_train_classif, y_train_classif)

    train_error = hc_model.score(scl_X_train_classif, y_train_classif)
    test_error = hc_model.score(scl_X_test, y_test)

    print("Train error is: %s; test error is: %s" % (train_error, test_error))
