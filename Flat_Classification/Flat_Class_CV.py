import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.linear_model
import sklearn.utils
import copy
import sklearn.cross_validation
import Flat_Classification.constants as ct

root = r"C:\Users\Vlad Dobru\Desktop\DataFinal\\"
split_root =r"C:\Users\Vlad Dobru\Documents\hierarchical_classification\Datasets\\all_categories_"
OUT = r"C:\Users\Vlad Dobru\Desktop\Foundations of Machine Learning\Project\Large Scale Classification\Results\\"

LEAVES = pd.read_pickle(root+"Pickles\Leaves_by_Category.p")
N_FOLDS = 3

def check_y(y, cat):
    cat_leaves = LEAVES.get(cat)
    y_series = pd.Series(y)

    return (y_series.isin(cat_leaves).product() == 1)


def check_x_y(x, y):
    return (x.shape[0] == len(y))


def throw_out_rare_labels(X, y, cv):
    y_series = pd.Series(y)
    val_counts = y_series.value_counts()
    less_than_allowed = val_counts[val_counts < cv].index

    print("Throwing out %s observations or %s of the sample" %
          (len(less_than_allowed), len(less_than_allowed) / len(y)))

    mask = ~ y_series.isin(less_than_allowed)

    return X[mask.values], y[mask.values]


def do_CV(cat, model, X, y, C_list=[0.001, 0.01, 0.1, 1, 10, 100, 1000], n_folds=3):

    scores = pd.Series(index=C_list)

    cvf = sklearn.cross_validation.KFold(n=len(y), n_folds=n_folds)

    for c in C_list:
        c_scores = []
        print("CV Fitting for C=%s" % c)

        for train, test in cvf:
            model_use = copy.deepcopy(model)
            model_use.C = c
            model_use.fit(X[train], y[train])
            score = model_use.score(X[test], y[test])
            del (model_use)
            print("CV score is %s" % score)
            c_scores.append(score)
        scores.loc[c] = pd.Series(c_scores).mean()

    pd.to_pickle(scores, root + "Pickles\\Flat_CV\\%s_%s_CV.p" % (cat, TYPE))

    scores.sort(ascending=False)
    best_c = scores.index[0]

    return best_c


def do_CV_for_category(cat, model_type):
    print("Fitting for category: %s" % cat)

    # Reading X pickles
    scl_X_train_classif = pd.read_pickle(root + "Pickles\\Scaled\\%s_X_scl_train_classif.p" % cat)
    scl_X_test = pd.read_pickle(root + "Pickles\\Scaled\\%s_X_scl_test_hier.p" % cat)

    # Reading y pickles
    y_train_classif = pd.read_pickle(root + "Pickles\\Scaled\\%s_y_train_classif.p" % cat)
    y_test = pd.read_pickle(root + "Pickles\\Scaled\\%s_y_test.p" % cat)

    # Checking that the y labels are fine
    assert (check_y(y_train_classif, cat))
    assert (check_y(y_test, cat))

    # Checking that the x and the ys match
    assert (check_x_y(scl_X_train_classif, y_train_classif))
    assert (check_x_y(scl_X_test, y_test))

    # Checking that the scaling is fine
    assert (scl_X_train_classif.max() == 1.0)

    # Throw out very rare classes
    scl_X_train_classif, y_train_classif = throw_out_rare_labels(
        scl_X_train_classif, y_train_classif, cv=N_FOLDS)

    # Run CV and output CV scores
    model = ct.MODELS.get(model_type)

    best_c = do_CV(cat, model, scl_X_train_classif, y_train_classif)

    return best_c

if __name__ == '__main__':

   TYPE='LR_OVR'
   best_CV = pd.Series(index=range(0, 11))

   for cat in range(0,11):
       best_CV.loc[cat] = do_CV_for_category(cat, model_type=TYPE)

   pd.to_pickle(best_CV, root + "Pickles\\Flat_CV\\ALL_%s_CV.p" % (cat, TYPE))