import pandas as pd
import numpy as np
import scipy.sparse
import Flat_Classification.constants as ct

def get_connect_matrix(cat):

    # Reading y pickle
    y = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_train_hier.p" % cat)

    spr_connect = scipy.sparse.lil_matrix((len(y), len(y)))
    y_unique = np.unique(y)

    for y_value in y_unique:
        mask = (y == y_value)
        for i in range(len(mask)):
            if (mask[i]):
                spr_connect[mask, i] = 1

    spr_connect = spr_connect.tocsr()

    return spr_connect

if __name__ == '__main__':

   for cat in range(11):

        # Reading y pickle
        y = pd.read_pickle(ct.ROOT + "Pickles\\Scaled\\%s_y_train_hier.p" % cat)

        print("Building connectivity matrix for category: %s" % cat)
        conn_matrix = get_connect_matrix(y)

        pd.to_pickle(conn_matrix, ct.ROOT + "Pickles\\Conn_Matrix\\%s_Conn_Matrix.p" % cat)