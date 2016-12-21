import pandas as pd
import copy
import Flat_Classification.constants as ct
import Flat_Classification.checking as ck

def fit_model(model, C, X_train, y_train, X_test, y_test):

    model_use = copy.deepcopy(model)
    model_use.C = C
    print(model_use.C, model_use.penalty)

    model_use.fit(X_train, y_train)
    train_acc = model_use.score(X_train, y_train)
    test_acc = model_use.score(X_test, y_test)

    return train_acc, test_acc

def fit_model(model, C, X_train, y_train, X_test, y_test):

    model_use = copy.deepcopy(model)
    model_use.C = C
    print(model_use.C, model_use.penalty)

    model_use.fit(X_train, y_train)
    train_acc = model_use.score(X_train, y_train)
    test_acc = model_use.score(X_test, y_test)

    del model_use

    return train_acc, test_acc

def fit_model_cat(model_type, C, cat):

    print("Fitting for category: %s" % cat)

    # Reading X pickles
    scl_X_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_X_scl_train_classif.p" % cat)
    scl_X_test = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_X_scl_test_hier.p" % cat)

    # Reading y pickles
    y_train_classif = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_train_classif.p" % cat)
    y_test = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_test.p" % cat)

    # Checking that the y labels are fine
    assert (ck.check_y(y_train_classif, cat))
    assert (ck.check_y(y_test, cat))

    # Checking that the x and the ys match
    assert (ck.check_x_y(scl_X_train_classif, y_train_classif))
    assert (ck.check_x_y(scl_X_test, y_test))

    # Checking that the scaling is fine
    assert (scl_X_train_classif.max() == 1.0)

    return fit_model(ct.MODELS[model_type], C,
                     scl_X_train_classif,
                     y_train_classif,
                     scl_X_test,
                     y_test
                     )


def get_best_C(cv_values):

    top = cv_values.sort_values(ascending=False).iloc[[0]]
    best_C = top.index[0]
    best_acc = top[best_C]

    return best_C, best_acc

if __name__ == '__main__':

    summ_stats = pd.DataFrame(index=range(0, 10),
                              columns=["LR_OVR: Best C", "LR_OVR: CV Accur", "LR_OVR: Train Accur",
                                       "LR_OVR: Test Accur",
                                       "SVM_OVR: Best C", "SVM_OVR: CV Accur", "SVM_OVR: Train Accur",
                                       "SVM_OVR: Test Accur",
                                       ])

    for model_type in ['SVM_OVR', 'LR_OVR']:

        print("Running CV for %s" % model_type)

        for cat in range(10):

            # Read the CV accuracies
            CV_accuracies = pd.read_pickle(ct.ROOT + "Pickles\\Flat_CV\\%s_%s_CV.p" % (cat, model_type))
            best_C, best_CV_acc = get_best_C(CV_accuracies)

            print("Best C and best CV accurancy:", best_C, best_CV_acc)

            summ_stats.ix[cat, "%s: Best C" % model_type] = best_C
            summ_stats.ix[cat, "%s: CV Accur" % model_type] = best_CV_acc

            train_acc, test_acc = fit_model_cat(model_type, best_C, cat)

            print("Train accur and test accur are:", train_acc, test_acc)
            #
            # pd.to_pickle(model_obj, ct.ROOT+"Pickles\\Flat_Final\\%s_%s_model.p" % (cat, model_type))
            # del(model_obj)

            summ_stats.ix[cat, "%s: Train Accur" % model_type] = train_acc
            summ_stats.ix[cat, "%s: Test Accur" % model_type] = test_acc

            pd.to_pickle(summ_stats, ct.ROOT+"Pickles\\Flat_Final\\Flat_Summ_Stats_%s.p" % cat)




