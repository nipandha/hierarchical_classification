import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.linear_model
import sklearn.utils
import copy
import sklearn.cross_validation

MODELS = {"LR_OVR": sklearn.linear_model.LogisticRegression(fit_intercept=True,
                                                            dual=True,
                                                            penalty='l2',
                                                            multi_class='ovr',
                                                            solver='liblinear',
                                                            tol=0.001),

          # Uses stochastic gradient descent - should consume less memory, but should be
          # slower. Also, solving it in the dual is not supported
          "LR_OVR_SAG": sklearn.linear_model.LogisticRegression(fit_intercept=True,
                                                                dual=False,
                                                                penalty='l2',
                                                                multi_class='ovr',
                                                                solver='sag',
                                                                tol=0.001,
                                                                max_iter=2500),

          "SVM_OVR": sklearn.svm.LinearSVC(fit_intercept=True,
                                           loss='hinge',
                                           dual=True,
                                           penalty='l2',
                                           multi_class='ovr',
                                           tol=0.001),
          }