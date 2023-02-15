import ot
from ot.utils import unif
from sklearn.cluster import SpectralClustering
import numpy as np


def sc(dataset: list, n_clu=2, seed=1):
    n_graph = len(dataset)
    M = np.zeros([n_graph, n_graph])
    for i in range(n_graph):
        for j in range(i + 1, n_graph):
            ni, nj = len(dataset[i]), len(dataset[j])
            pi, pj = unif(ni), unif(nj)
            M[i, j] = ot.gromov.gromov_wasserstein2(dataset[i], dataset[j], pi, pj)
            M[j, i] = M[i, j]
    print("GW matrix constructed")
    clustering = SpectralClustering(n_clusters=n_clu, assign_labels='discretize', affinity="precomputed",
                                    random_state=seed).fit(M)
    return clustering.labels_
