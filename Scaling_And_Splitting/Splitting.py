import Flat_Classification.constants as ct
import pandas as pd
import numpy as np
import numpy.random as np_random
import sklearn
import sklearn.preprocessing
import sklearn.datasets
import sklearn.utils
import copy

split_root =r"C:\Users\Vlad Dobru\Documents\hierarchical_classification\Datasets\\all_categories_"
OUT = r"C:\Users\Vlad Dobru\Desktop\Foundations of Machine Learning\Project\Large Scale Classification\Results\\"

# Split percentages
TRAIN = 0.7
TEST = 0.3
TRAIN_HIER = 0.4

def load_and_pickle_leaves():

    LEAVES = {}

    for i in range(0, 10):
        cat_leaves = pd.read_table(split_root + str(i), header=None)
        LEAVES[i] = cat_leaves[0].tolist()

    pd.to_pickle(LEAVES, ct.ROOT + "Pickles\Leaves_by_Category.p")

def get_slicing_index(y, cat_leaves):

    return pd.Series(y).isin(cat_leaves).values

def get_num_features(X):

    return len(np.unique(X.nonzero()[1]))

def get_num_classes(y, cat_leaves):

    return get_slicing_index(np.unique(y), cat_leaves).sum()

def check_split(y_first, y_second):
    return (len(np.unique(y_first)) == len(np.unique(y_second)))

def do_split(X, y, split_percent, check=False):
    X = copy.deepcopy(X)
    y = copy.deepcopy(y)

    cutoff = int(np.floor(len(y) * split_percent))

    count = 0;

    while True:
        count += 1
        X_first = X[:cutoff]
        y_first = y[:cutoff]

        X_second = X[cutoff:]
        y_second = y[cutoff:]

        if (check_split(y_first, y_second)):
            break
        elif (count > 50):
            print("Cannot ensure that we have the same number of labels!")
            break
        elif check:
            print("Trying the split again!")
            X, y = sklearn.utils.shuffle(X, y)
        else:
            break

    return X_first, y_first, X_second, y_second

def split_pickle_and_output_stats():

    # Load Data
    X, y = sklearn.datasets.load_svmlight_file(f=ct.ROOT+"trainDMOZ.txt")

    # Randomly shullfe the data
    X, y = sklearn.utils.shuffle(X, y, random_state=0)

    # Load the Leaves

    LEAVES = pd.read_pickle(ct.ROOT + "Pickles\Leaves_by_Category.p")

    # Initialize summary stats

    summ_stats = pd.DataFrame(columns=["N", "N Train for Hierarchy", "N Train for Classification", "N Test",
                                       "# of classes", "# of features"])

    # Do the splitting

    for cat, cat_leaves in LEAVES.items():
        print("We are on category: ", cat)

        # Split by cat
        X_cat = X[get_slicing_index(y, cat_leaves)]
        y_cat = y[get_slicing_index(y, cat_leaves)]

        # Insert summary stats
        summ_stats.loc[cat, "N"] = len(y_cat)
        summ_stats.loc[cat, "# of classes"] = get_num_classes(y, cat_leaves)
        summ_stats.loc[cat, "# of features"] = get_num_features(X_cat)

        # Split into train and test
        print("Split into train and test!")
        X_train, y_train, X_test, y_test = do_split(X_cat, y_cat, TRAIN, check=False)
        summ_stats.loc[cat, "N Test"] = len(y_test)

        # Output test
        pd.to_pickle(X_test, ct.ROOT + "Pickles\\Unscaled\\%s_X_test.p" % cat)
        pd.to_pickle(y_test, ct.ROOT + "Pickles\\Unscaled\\%s_y_test.p" % cat)

        # Split into "train for hierarchy" and "train for classification"
        print("Split training sample into train for hier and train for classif")
        X_train_hier, y_train_hier, X_train_classif, y_train_classif = do_split(X_train, y_train, TRAIN_HIER,
                                                                                check=True)

        summ_stats.loc[cat, "N Train for Hierarchy"] = len(y_train_hier)
        summ_stats.loc[cat, "N Train for Classification"] = len(y_train_classif)

        # Output train for hier
        pd.to_pickle(X_train_hier, ct.ROOT + "Pickles\\Unscaled\\%s_X_train_hier.p" % cat)
        pd.to_pickle(y_train_hier, ct.ROOT + "Pickles\\Unscaled\\%s_y_train_hier.p" % cat)

        # Output train for classif
        pd.to_pickle(X_train_classif, ct.ROOT + "Pickles\\Unscaled\\%s_X_train_classif.p" % cat)
        pd.to_pickle(y_train_classif, ct.ROOT + "Pickles\\Unscaled\\%s_y_train_classif.p" % cat)

    return summ_stats

