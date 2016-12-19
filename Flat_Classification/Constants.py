import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.cross_validation

ROOT = r"C:\Users\Vlad Dobru\Desktop\DataFinal\\"
SLIT_ROOT =r"C:\Users\Vlad Dobru\Documents\hierarchical_classification\Datasets\\all_categories_"
OUT = r"C:\Users\Vlad Dobru\Desktop\Foundations of Machine Learning\Project\Large Scale Classification\Results\\"

LEAVES = pd.read_pickle(ROOT+"Pickles\Leaves_by_Category.p")

MODELS = {"LR_OVR": sklearn.linear_model.LogisticRegression(fit_intercept=True,
                                                            dual=True,
                                                            penalty='l2',
                                                            multi_class='ovr',
                                                            solver='liblinear',
                                                            tol=0.02),

          # Uses stochastic gradient descent - should consume less memory, but should be
          # slower. Also, solving it in the dual is not supported
          "LR_OVR_SAG": sklearn.linear_model.LogisticRegression(fit_intercept=True,
                                                                dual=False,
                                                                penalty='l2',
                                                                multi_class='ovr',
                                                                solver='sag',
                                                                tol=0.02,
                                                                max_iter=1500),

          # Best theoretical model, but very computationally expensive
          "LR_OVO_SAG": sklearn.linear_model.LogisticRegression(fit_intercept=True,
                                                                dual=False,
                                                                penalty='l2',
                                                                multi_class='multinomial',
                                                                solver='newton-cg',
                                                                tol=0.02,
                                                                max_iter=1500),

          "SVM_OVR": sklearn.svm.LinearSVC(fit_intercept=True,
                                           loss='hinge',
                                           dual=True,
                                           penalty='l2',
                                           multi_class='ovr',
                                           tol=0.02),
          }