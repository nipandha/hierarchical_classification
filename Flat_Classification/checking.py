import pandas as pd
import Flat_Classification.constants as ct

def check_y(y, cat):
    cat_leaves = ct.LEAVES.get(cat)
    y_series = pd.Series(y)

    return (y_series.isin(cat_leaves).product() == 1)


def check_x_y(x, y):
    return (x.shape[0] == len(y))