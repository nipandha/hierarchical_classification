import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.linear_model
import sklearn.utils

import Hierarchical_Classification.hc_model as hcm
import sklearn.cross_validation
import Flat_Classification.constants as ct

N_FOLDS = 3
C_LIST = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4]

def do_CV(cat, model, X, y, model_type, C_list=C_LIST, n_folds=3):

    scores = pd.Series(index=C_list)

    cvf = sklearn.cross_validation.KFold(n=len(y), n_folds=n_folds)

    for c in C_list:
        c_scores = []
        print("CV Fitting for C=%s" % c)

        for train, test in cvf:
            hc_model = hcm.HierarchyClassification(cat, model, C=c)
            print("Checking that C=%s" % hc_model.C)

            hc_model.fit(X[train], y[train])
            score = hc_model.score(X[test], y[test])
            del (hc_model)

            print("CV score is %s" % score)
            c_scores.append(score)

            break;

        scores.loc[c] = pd.Series(c_scores).mean()

    pd.to_pickle(scores, ct.ROOT + "Pickles\\Fitted_Hierarchy_CV\\%s_%s_CV.p" % (cat, model_type))

    scores.sort(ascending=False)
    best_c = scores.index[0]

    return best_c

def do_CV_for_category(cat, model_type):
    print("Fitting for category: %s" % cat)

    # Reading X pickles
    scl_X_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_X_scl_train_classif.p" % cat)

    # Reading y pickles
    y_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_train_classif.p" % cat)

    # Run CV and output CV scores
    model = ct.MODELS.get(model_type)

    best_c = do_CV(cat, model, scl_X_train_classif, y_train_classif, model_type)

    return best_c

if __name__ == '__main__':

   best_CV = pd.Series(index=range(10))

   for model_type in ['LR_OVR', 'SVM_OVR']:

       print("Running CV for %s" % model_type)

       for cat in [4]:
            best_CV.loc[cat] = do_CV_for_category(cat, model_type)

       pd.to_pickle(best_CV, ct.ROOT + "Pickles\\Flat_CV\\ALL_%s_CV.p" % model_type)