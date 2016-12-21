import pandas as pd
import copy
import sklearn.cluster
import Flat_Classification.constants as ct

CLUST_MODELS = {
                "Cos_Complete_2": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=2, linkage='complete',
                    compute_full_tree=True, affinity='cosine'),
                "Cos_Complete_3": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=3, linkage='complete',
                    compute_full_tree=True, affinity='cosine'),
                "Cos_Average_2": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=2, linkage='average',
                    compute_full_tree=True, affinity='cosine'),
                "Cos_Average_3": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=3, linkage='average',
                    compute_full_tree=True,  affinity='cosine'),
                "Complete_2": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=2, linkage='complete',
                    compute_full_tree=True),
                "Complete_3": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=3, linkage='complete',
                    compute_full_tree=True),
                "Ward_2": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=2, linkage='ward',
                    compute_full_tree=True),
                "Ward_3": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=3, linkage='ward',
                    compute_full_tree=True),
                "Average_2": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=2, linkage='average',
                    compute_full_tree=True),
                "Average_3": sklearn.cluster.AgglomerativeClustering(
                    n_clusters=3, linkage='average',
                    compute_full_tree=True),
                }

def fit_and_output(model, avg_X, path_stub):

    model_use = copy.deepcopy(model)
    terminal_nodes = model_use.fit_predict(avg_X)
    children = model_use.children_

    # Outputting the model
    pd.to_pickle(model_use, path_stub+"fitted_model.p")

    # Outputting the children
    pd.DataFrame(children).to_csv(
                    path_stub + "children.csv",
                    header=False,
                    index=False)

    # Outputting the terminal nodes
    pd.Series(terminal_nodes).to_csv(
                    path_stub + "terminal_nodes.csv",
                    index=False)

    return None

if __name__ == '__main__':

    for model_type, model in CLUST_MODELS.items():
        print("Running the following clustering algo: %s" % model_type)

        for cat in range(11):
            print("Running algo for category: %s" % cat)
            path_stub = ct.ROOT+"Pickles\\Hierarchy\\"+model_type+("\\%s_" % cat)
            # Get Compressed features
            avg_X = pd.read_pickle(ct.ROOT+"Pickles\\Avg_Feature_Matrices\\%s_Avg_Feat_Matrix.p" % cat)
            fit_and_output(model, avg_X, path_stub)
            del avg_X