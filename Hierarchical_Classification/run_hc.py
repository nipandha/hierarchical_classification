import pandas as pd

import Flat_Classification.constants as ct
import Flat_Classification.flat_class as fc
import Hierarchical_Classification.hc_model as hcm
import Hierarchical_Classification.hierarchy as h

def fit_model_cat(model_type, best_C, cat):

    # Reading X pickles
    scl_X_train_classif = pd.read_pickle(h.ROOT + "Scaled\\%s_X_scl_train_classif.p" % cat)
    scl_X_test = pd.read_pickle(h.ROOT + "Scaled\\%s_X_scl_test_hier.p" % cat)

    # Reading y pickles
    y_train_classif = pd.read_pickle(h.ROOT + "Scaled\\%s_y_train_classif.p" % cat)
    y_test = pd.read_pickle(h.ROOT + "Scaled\\%s_y_test.p" % cat)

    hc_model = hcm.HierarchyClassification(cat, ct.MODELS[model_type], C=best_C)
    hc_model.fit(scl_X_train_classif, y_train_classif)

    train_acc = hc_model.score(scl_X_train_classif, y_train_classif)
    test_acc = hc_model.score(scl_X_test, y_test)

    return train_acc, test_acc

if __name__ == '__main__':
    summ_stats = pd.DataFrame(index=range(0, 10),
                              columns=["LR_OVR: Best C", "LR_OVR: CV Accur", "LR_OVR: Train Accur",
                                       "LR_OVR: Test Accur",
                                       "SVM_OVR: Best C", "SVM_OVR: CV Accur", "SVM_OVR: Train Accur",
                                       "SVM_OVR: Test Accur",
                                       ])

for model_type in ['SVM_OVR', 'LR_OVR']:

    print("Running CV for %s" % model_type)

    for cat in range(0, 10):
        # Read the CV accuracies
        CV_accuracies = pd.read_pickle(ct.ROOT + "Pickles\\Fitted_Hierarchy_CV\\%s_%s_CV.p" % (cat, model_type))
        best_C, best_CV_acc = fc.get_best_C(CV_accuracies)

        print("Best C and best CV accurancy:", best_C, best_CV_acc)

        summ_stats.ix[cat, "%s: Best C" % model_type] = best_C
        summ_stats.ix[cat, "%s: CV Accur" % model_type] = best_CV_acc

        train_acc, test_acc = fit_model_cat(model_type, best_C, cat)

        print("Train accur and test accur are:", train_acc, test_acc)

        summ_stats.ix[cat, "%s: Train Accur" % model_type] = train_acc
        summ_stats.ix[cat, "%s: Test Accur" % model_type] = test_acc

    pd.to_pickle(summ_stats, ct.ROOT + "Pickles\\Fitted_Hierarchy\\HC_Summ_Stats.p")


